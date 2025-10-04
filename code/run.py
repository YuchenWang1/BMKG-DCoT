import os
import logging
import argparse
import datetime
import warnings

import jsonlines
from tqdm import tqdm

from GraphAgent import GraphAgent
from tools.retriever import NODE_TEXT_KEYS
from graph_prompts import graph_agent_prompt, graph_agent_prompt_zeroshot

# Ignore deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def remove_fewshot(prompt: str) -> str:
    """
    Removes the few-shot examples from a given prompt string.

    Args:
        prompt (str): The full prompt content including few-shot examples.

    Returns:
        str: The prompt with few-shot examples removed.
    """
    prefix = prompt[-1].content.split('下面是一些示例：')[0]
    suffix = prompt[-1].content.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n' + suffix.strip('\n').strip()


def main():
    """
    Main function to run the graph agent experiment.
    """
    parser = argparse.ArgumentParser(description="Run Graph Agent experiments.")
    parser.add_argument("--dataset", type=str, default="bridge", help="Name of the dataset.")
    parser.add_argument("--path", type=str, default="../data", help="Path to the data directory.")
    parser.add_argument("--save_file", type=str, default="../result", help="Path to save the results file.")
    parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Embedding model name.")
    parser.add_argument("--faiss_gpu", type=bool, default=False, help="Whether to use GPU for Faiss.")
    parser.add_argument("--embed_cache", type=bool, default=True, help="Whether to use cached embeddings.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps for the agent.")
    parser.add_argument("--ref_dataset", type=str, default=None, help="Reference dataset for few-shot examples.")
    parser.add_argument("--llm_version", type=str, default="glm-4-flash", help="LLM version to use.")
    parser.add_argument("--zero_shot", action='store_true', help="Enable zero-shot prompting.")
    args = parser.parse_args()

    # Configure paths
    args.embed_cache_dir = args.path
    args.graph_dir = os.path.join(args.path, "graph.json")
    args.data_dir = os.path.join(args.path, "data.json")
    args.node_text_keys = NODE_TEXT_KEYS[args.dataset]
    args.ref_dataset = args.dataset if not args.ref_dataset else args.ref_dataset

    # Validate LLM version
    supported_llms = [
        'gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k',
        'glm-4-flash', "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Llama-2-13b-chat-hf", 'models/gemini-2.0-flash',
        'llama3-8b-hf', 'qwen'
    ]
    assert args.llm_version in supported_llms, f"Unsupported LLM version: {args.llm_version}"

    # Load dataset
    with open(args.data_dir, 'r') as f:
        contents = [item for item in jsonlines.Reader(f)]

    # Prepare output directory and logging
    output_file_path = args.save_file
    parent_folder = os.path.dirname(output_file_path)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)

    logs_dir = os.path.join(parent_folder, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Initialize agent
    agent_prompt = graph_agent_prompt_zeroshot if args.zero_shot else graph_agent_prompt
    agent = GraphAgent(args, agent_prompt)

    # Initialize result lists
    correct_logs = []
    halted_logs = []
    incorrect_logs = []
    generated_text = []

    # Main processing loop
    for item in tqdm(contents):
        agent.run(item['question'], item['answer'])
        print(f'Ground Truth Answer: {agent.key}')
        print('---------')

        # Create log entry
        log = f"Question: {item['question']}\n"
        prompt_content = agent._build_agent_prompt_with_path(agent.beam_paths[0], is_reasoning=False)
        if not args.zero_shot:
            log += remove_fewshot(prompt_content) + f'\nCorrect answer: {agent.key}\n\n'
        else:
            log += prompt_content[-1].content + f'\nCorrect answer: {agent.key}\n\n'

        with open(os.path.join(logs_dir, item['qid'] + '.txt'), 'w', encoding='utf-8') as f:
            f.write(log)

        # Aggregate results from all beam paths
        for path in agent.beam_paths:
            if path.finished:
                if agent.is_correct(path.answer, agent.key):
                    correct_logs.append(log)
                else:
                    incorrect_logs.append(log)
            else:
                halted_logs.append(log)

            generated_text.append({
                "question": item["question"],
                "model_answer": path.answer,
                "gt_answer": item['answer'],
                "scratchpad": path.scratchpad,
                "score": path.score
            })

    # Write results to file
    with jsonlines.open(output_file_path, 'w') as writer:
        writer.write_all(generated_text)

    # Print summary
    print(
        f'Finished Trial {len(contents)}, Correct: {len(correct_logs)}, Incorrect: {len(incorrect_logs)}, Halted: {len(halted_logs)}')


if __name__ == '__main__':
    main()