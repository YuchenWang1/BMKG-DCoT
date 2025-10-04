import glob
import os
import sys
import json
import pickle
from contextlib import nullcontext
from typing import Dict, List
import logging

import faiss
import numpy as np
import torch
from tqdm import tqdm
import sentence_transformers
from sentence_transformers import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NODE_TEXT_KEYS = {
    'bridge': {'构件编号': ['name'], '构件部位': ['name'], '病害位置': ['name'], '病害': ['name'], '构件': ['name'],
               '桥梁名称': ['name'], '维护措施': ['name']}
    }


class Retriever:

    def __init__(self, args, graph, cache=True, cache_dir=None):
        logger.info("Initializing retriever")
        logger.info("初始化检索器")

        self.use_gpu = args.faiss_gpu
        self.node_text_keys = args.node_text_keys
        self.model_name = args.embedder_name
        self.model = sentence_transformers.SentenceTransformer(args.embedder_name)
        self.graph = graph
        self.cache = args.embed_cache
        self.cache_dir = args.embed_cache_dir

        self.reset()

    def reset(self):

        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]

        if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')):
            embeds, self.doc_lookup, self.doc_type = pickle.load(
                open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self.multi_gpu_infer(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type],
                        open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'wb'))

        self.init_index_and_add(embeds)

    def process_graph(self):
        docs = []
        ids = []
        meta_type = []

        for node_type_key in self.graph.keys():
            node_type = node_type_key.split('_nodes')[0]
            logger.info(f'为{node_type}加载文本')
            for nid in tqdm(self.graph[node_type_key]):
                docs.append(str(self.graph[node_type_key][nid]['features'][self.node_text_keys[node_type][0]]))
                ids.append(nid)
                meta_type.append(node_type)
        return docs, ids, meta_type

    def multi_gpu_infer(self, docs):
        pool = self.model.start_multi_process_pool()
        embeds = self.model.encode_multi_process(docs, pool)
        return embeds

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def init_index_and_add(self, embeds):

        logger.info("初始化索引...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)

        if self.use_gpu:
            self._move_index_to_gpu()

    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model, args):
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []

    def search_single(self, query, topk: int = 10):

        if self.index is None:
            raise ValueError("Index is not initialized")

        query_embed = self.model.encode(query, show_progress_bar=False)

        D, I = self.index.search(query_embed[None, :], topk)
        original_indice = np.array(self.doc_lookup)[I].tolist()[0][0]
        original_type = np.array(self.doc_type)[I].tolist()[0][0]

        return original_indice, self.graph[f'{original_type}_nodes'][original_indice]

    def search_batch(self, queries: List[str], topk: int = 10):
        if self.index is None:
            raise ValueError("Index is not initialized")

        query_embeds = self.model.encode(queries, show_progress_bar=False)
        D, I = self.index.search(query_embeds, topk)
        results = []
        for d, i_indices in zip(D, I):
            result = []
            for idx in i_indices:
                original_indice = self.doc_lookup[idx]
                original_type = self.doc_type[idx]
                node = self.graph[f'{original_type}_nodes'][original_indice]
                result.append((original_indice, node))
            results.append(result)
        return results

    def compute_similarity(self, text1: str, text2: str) -> float:

        emb1 = self.model.encode([text1], show_progress_bar=False)
        emb2 = self.model.encode([text2], show_progress_bar=False)
        sim = float(util.cos_sim(emb1, emb2)[0][0])
        return sim