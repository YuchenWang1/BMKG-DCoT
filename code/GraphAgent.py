# 该 Python 脚本定义了一个名为 GraphAgent 的人工智能代理。
# 其核心功能是利用知识图谱来回答与桥梁维护相关的问题。
# 该代理采用束搜索（Beam Search）算法来探索不同的推理路径，并与多种大型语言模型（LLM）
# 如 Gemini, GPT, ZhipuAI 等进行交互，以执行推理和行动。
# 代理能够执行一系列预定义的操作，例如查找相似的桥梁、构件和病害，
# 并最终查询相应的维护措施，旨在为桥梁工程师提供决策支持。

import re
import string
import os
import json
import logging
import ast
import heapq
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import openai
import zhipuai
from langchain_community.chat_models import ChatOpenAI, ChatZhipuAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from graph_prompts import GRAPH_DEFINITION
from graph_fewshots import EXAMPLES
from tools import graph_funcs_maintenance, retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(order=True)
class BeamPath:
    """
    Represents a single path in the beam search. Each path has a score,
    a set of current nodes, a scratchpad for reasoning, and a final answer.
    """
    score: float
    current_nodes: Dict[str, Any] = field(compare=False)  # Information about nodes in the current step.
    scratchpad: str = field(compare=False)  # Stores the reasoning process.
    answer: Any = field(compare=False, default=None)  # Stores the final answer.
    finished: bool = field(compare=False, default=False)  # Flag indicating if the path is complete.


class GraphAgent:
    """
    An agent that uses a knowledge graph and LLMs to answer questions
    related to bridge maintenance using a beam search algorithm.
    """

    def __init__(self,
                 args,
                 agent_prompt,
                 beam_width=5
                 ) -> None:
        """
        Initializes the GraphAgent.

        Args:
            args: Configuration arguments.
            agent_prompt: The prompt template for the agent.
            beam_width (int): The width of the beam for the search algorithm.
        """
        self.max_steps = args.max_steps
        self.agent_prompt = agent_prompt
        self.examples = EXAMPLES[args.ref_dataset]
        self.beam_width = beam_width
        self.llm_version = args.llm_version

        # Configuration for calling Gemini via an OpenAI-compatible endpoint.
        if self.llm_version in ['models/gemini-2.0-flash']:
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=300,
                model_name=self.llm_version,
                openai_api_key="YOUR_GEMINI_API_KEY",
                openai_api_base="https://generativelanguage.googleapis.com/v1beta",
                model_kwargs={"stop": "\n"},
            )
            self.enc = None

        # Configuration for Qwen model via Dashscope.
        elif self.llm_version == "qwen":
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=300,
                model_name="qwen3-8b",
                openai_api_key="YOUR_QWEN_API_KEY",
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_kwargs={
                    "stop": "\n",
                    "extra_body": {
                        "enable_thinking": False
                    }
                }
            )
            self.enc = None

        # Configuration for official OpenAI models.
        elif self.llm_version in ['gpt-5']:
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=300,
                openai_api_key="YOUR_OPENAI_API_KEY",
                model_name=self.llm_version,
                model_kwargs={"stop": "\n"},
            )
            self.enc = None

        # Configuration for ZhipuAI models.
        elif self.llm_version in ["glm-4-flash"]:
            self.llm = ChatZhipuAI(
                temperature=0,
                max_tokens=300,
                model_name=self.llm_version,
                api_key='YOUR_GLM_API_KEY',
                model_kwargs={"stop": "\n"},
                request_timeout=30
            )
            logger.warning("Zhipu AI model does not support tiktoken encoder; using simple word count.")
            self.enc = None
        else:
            raise ValueError("The provided llm_version is not correct.")

        self.graph_definition = GRAPH_DEFINITION[args.dataset]
        self.load_graph(args.graph_dir)
        self.graph_funcs = graph_funcs_maintenance.graph_funcs(self.graph)
        self.node_retriever = retriever.Retriever(args, self.graph)
        self.__reset_agent()

    def load_graph(self, graph_dir):
        """Loads the graph from a specified JSON file."""
        logger.info('Loading graph...')
        self.graph = json.load(open(graph_dir, encoding='utf-8'))

    def run(self, question, answer, reset=True) -> None:
        """
        Runs the agent to answer a question.

        Args:
            question (str): The question to answer.
            answer (str): The ground truth answer for evaluation.
            reset (bool): Whether to reset the agent's state before running.
        """
        if reset:
            self.__reset_agent()
        self.question = question
        self.key = answer

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        """Performs a single step of the beam search across all active paths."""
        new_beam_paths = []
        for path in self.beam_paths:
            if path.finished:
                # If a path is finished, keep it for the next iteration.
                new_beam_paths.append(path)
                continue

            # --- Reasoning Step ---
            path.scratchpad += f'\nThought {self.step_n}:'
            reasoning = self.prompt_agent(path, is_reasoning=True)
            path.scratchpad += ' ' + reasoning
            print(reasoning)

            # --- Action Step ---
            path.scratchpad += f'\nAction {self.step_n}:'
            action = self.prompt_agent(path, is_reasoning=False)
            path.scratchpad += ' ' + action
            print(action)

            # Parse the generated action.
            action_list = get_action_list(action)
            for tmp_action in action_list:
                try:
                    parsed = parse_action(tmp_action)
                    if parsed is None:
                        raise ValueError("Action parsing failed")
                    action_type, argument = parsed
                except:
                    path.scratchpad += 'The generated action is invalid.'
                    continue

                # --- Handle Different Action Types ---
                if action_type == '完成':  # Finish
                    try:
                        path.answer = eval(argument)
                    except:
                        path.answer = argument

                    if isinstance(path.answer, list):
                        aggregated_answer = "; ".join(path.answer)
                    else:
                        aggregated_answer = path.answer

                    if self.is_correct(aggregated_answer, self.key):
                        path.scratchpad += 'Correct answer.'
                        path.score += 1.0  # Increase score for correct answer.
                    else:
                        path.scratchpad += 'Incorrect answer.'
                    path.finished = True
                    new_beam_paths.append(path)
                    continue

                elif action_type == '筛选相似桥梁':  # Filter Similar Bridges
                    input_bridge_type = remove_quotes(argument)
                    try:
                        all_bridges = self.graph_funcs.get_all_nodes_by_type('桥梁名称')

                        similar_bridges = []
                        for b in all_bridges:
                            candidate_text = self.graph_funcs.merge_node_info_bridge(b)
                            sim = self.node_retriever.compute_similarity(input_bridge_type, candidate_text)
                            if sim > 0.7:
                                similar_bridges.append((b, sim))

                        similar_bridges.sort(key=lambda x: x[1], reverse=True)
                        top_similar_bridges = similar_bridges[:self.beam_width]

                        similar_bridge_ids = [b[0] for b in top_similar_bridges]
                        path.current_nodes['相似桥梁'] = similar_bridge_ids
                        path.scratchpad += f"Found similar bridge IDs: {similar_bridge_ids}"
                    except Exception as e:
                        path.scratchpad += f"Error while filtering similar bridges: {str(e)}."

                elif action_type == '筛选相似结构构件':  # Filter Similar Structural Components
                    args_list = argument.split(', ', 1)

                    if len(args_list) != 2:
                        path.scratchpad += "Incorrect number of arguments for filtering components. Expected [component_name, similar_bridge_ids]"
                        continue
                    comp_name = remove_quotes(args_list[0])
                    similar_bridges_str = args_list[1]
                    try:
                        similar_bridges_str = re.sub(r'[()\[\]\'\\]', '', similar_bridges_str).strip()
                        similar_bridges = ast.literal_eval(similar_bridges_str) if isinstance(similar_bridges_str,
                                                                                              str) else similar_bridges_str

                        similar_components_all = []
                        for bid in similar_bridges:
                            bid_str = str(bid)
                            result = self.graph_funcs.find_nodes_in_paths_with_similarity(bid_str, comp_name,
                                                                                          similarity_threshold=0.7)
                            matched_nodes = [nid for name, nid in result['exact']] + [nid for name, nid in
                                                                                      result['similar']]
                            similar_components_all.extend(matched_nodes)

                        similar_components_all = list(set(similar_components_all))

                        path.current_nodes['相似构件'] = similar_components_all
                        path.scratchpad += f"Found similar component IDs: {similar_components_all}"
                    except Exception as e:
                        path.scratchpad += f"Error while filtering similar components: {str(e)}."

                elif action_type == '筛选相似病害':  # Filter Similar Diseases
                    args_list = split_args(argument)
                    logger.info(f"Output: {args_list}")

                    if len(args_list) != 3:
                        path.scratchpad += "Incorrect number of arguments for filtering diseases. Expected [disease_name, disease_description, similar_component_ids]\n"
                        logger.warning("Incorrect number of arguments for filtering diseases.")
                        continue

                    disease_name = remove_quotes(args_list[0])
                    target_disease_text = remove_quotes(args_list[1])
                    similar_components_str = standardize_quotes(args_list[2])

                    try:
                        similar_components = ast.literal_eval(
                            similar_components_str) if similar_components_str.startswith('[') else [
                            similar_components_str]
                        similar_components = [str(comp_id) for comp_id in similar_components]

                        similar_disease_ids = []
                        disease_details = []
                        for comp_id in similar_components:
                            result = self.graph_funcs.find_nodes_in_paths_with_similarity(comp_id, disease_name,
                                                                                          similarity_threshold=0.65)
                            candidate_diseases = [nid for name, nid in result['exact']] + [nid for name, nid in
                                                                                           result['similar']]

                            if not candidate_diseases:
                                logger.warning(f"No related diseases found for component ID {comp_id}.")
                                continue

                            for cdid in candidate_diseases:
                                cd_text = self.graph_funcs.merge_node_info(cdid, comp_id, max_depth=4)

                                if "没有找到路径" in cd_text:  # "Path not found"
                                    logger.warning(f"Invalid description for disease ID {cdid}, skipping.")
                                    continue

                                sim = self.node_retriever.compute_similarity(target_disease_text, cd_text)
                                logger.info(
                                    f"Comparing target '{target_disease_text}' with component {comp_id}'s disease '{cd_text}'. Similarity: {sim}.")

                                is_similar = self.assess_similarity(target_disease_text, cd_text)
                                logger.info(f"LLM similarity assessment: {'Similar' if is_similar else 'Not similar'}.")

                                if is_similar or sim >= 0.65:
                                    similar_disease_ids.append(cdid)
                                    disease_details.append({
                                        '构件ID': comp_id,
                                        '病害ID': cdid,
                                        '病害描述': cd_text,
                                        '相似度': sim,
                                        'AI判断是否相似': is_similar
                                    })

                        if disease_details:
                            disease_details.sort(key=lambda x: x['相似度'], reverse=True)
                            most_similar_disease = disease_details[0]
                            path.current_nodes['最相似病害'] = [most_similar_disease['病害ID']]
                            path.scratchpad += f"Found most similar disease node ID: {most_similar_disease['病害ID']}\n"
                            logger.info(f"Found most similar disease node ID: {most_similar_disease['病害ID']}")
                            print(
                                f"Most similar disease: {most_similar_disease['病害描述']}, ID: {most_similar_disease['病害ID']}, Component ID: {most_similar_disease['构件ID']}, Similarity: {most_similar_disease['相似度']}")
                            logger.info(
                                f"Most similar disease: {most_similar_disease['病害描述']}, ID: {most_similar_disease['病害ID']}, Component ID: {most_similar_disease['构件ID']}, Similarity: {most_similar_disease['相似度']}")
                        else:
                            path.scratchpad += "No similar diseases found. Check argument count for the action.\n"
                            logger.warning("No similar diseases found.")

                    except Exception as e:
                        path.scratchpad += f"Error while filtering similar diseases: {str(e)}.\n"
                        logger.error(f"Error while filtering similar diseases: {str(e)}.")

                elif action_type == '维护措施查询':  # Query Maintenance Measures
                    disease_ids_str = remove_quotes(argument)
                    try:
                        disease_ids = ast.literal_eval(disease_ids_str) if disease_ids_str.startswith('[') else [
                            disease_ids_str]
                        disease_ids = [str(did) for did in disease_ids]
                        all_measures = []
                        for did in disease_ids:
                            measures = self.graph_funcs.get_maintenance_measures(did)
                            if measures:
                                all_measures.append(measures)
                                logger.info(f"Maintenance measures for disease ID {did}: {all_measures}")
                            else:
                                path.scratchpad += f"No maintenance measures found for disease ID {did}.\n"
                                logger.warning(f"No maintenance measures found for disease ID {did}.")

                        if all_measures:
                            path.current_nodes['维护措施'] = all_measures
                            path.scratchpad += f"Found maintenance measures: {all_measures}\n"
                            logger.info(f"Found maintenance measures: {all_measures}")
                        else:
                            path.scratchpad += "No maintenance measures were found.\n"
                            logger.warning("No maintenance measures were found.")

                    except Exception as e:
                        path.scratchpad += f"Error during maintenance measure query: {str(e)}.\n"
                        logger.error(f"Error during maintenance measure query: {str(e)}.")

                else:
                    path.scratchpad += 'Invalid operation. Valid operations include: 筛选相似桥梁[...], 筛选相似结构构件[...], 筛选相似病害[...], 维护措施查询[...].'

            if not path.finished:
                new_beam_paths.append(path)

        # Select the top K paths for the next step.
        self.beam_paths = heapq.nsmallest(self.beam_width, new_beam_paths)
        self.step_n += 1

    def prompt_agent(self, path: BeamPath, is_reasoning: bool = True) -> str:
        """
        Prompts the LLM for either a reasoning step or an action step.

        Args:
            path (BeamPath): The current beam search path.
            is_reasoning (bool): True to prompt for reasoning, False for action.

        Returns:
            str: The LLM's response.
        """
        if is_reasoning:
            messages = self._build_agent_prompt_with_path(path, is_reasoning=True)
        else:
            messages = self._build_agent_prompt_with_path(path, is_reasoning=False)

        try:
            response = self.llm(messages)
            return gpt_format_step(response)
        except Exception as e:
            logger.error(f"Error during LLM call: {e}")
            return f"Error: {e}"

    def _build_agent_prompt_with_path(self, path: BeamPath, is_reasoning: bool) -> List[Dict[str, str]]:
        """Constructs the full prompt for the LLM based on the current path."""
        return self.agent_prompt.format_messages(
            examples=self.examples,
            question=self.question,
            scratchpad=path.scratchpad,
            graph_definition=self.graph_definition
        )

    def is_correct(self, answer, key) -> bool:
        """Checks if the generated answer matches the key using exact match."""
        return EM(str(answer), str(key))

    def is_finished(self) -> bool:
        """Checks if all paths in the beam search have finished."""
        return all([path.finished for path in self.beam_paths])

    def is_halted(self) -> bool:
        """
        Checks if the agent should halt due to exceeding max steps or token length.
        """
        for path in self.beam_paths:
            if self.enc:
                prompt_length = len(self.enc.encode(path.scratchpad))
            else:
                prompt_length = len(path.scratchpad)

            if (self.step_n > self.max_steps) or (prompt_length > 10000):
                return True
        return False

    def count_tokens_zhipu(self, prompt: str) -> int:
        """A simple token counter for Zhipu AI models."""
        return len(prompt.split())

    def __reset_agent(self) -> None:
        """Resets the agent to its initial state for a new question."""
        self.step_n = 1
        self.finished = False
        self.beam_paths: List[BeamPath] = [
            BeamPath(score=0.0, current_nodes={}, scratchpad="")
        ]

    def set_qa(self, question: str, key: str) -> None:
        """Sets the current question and answer key."""
        self.question = question
        self.key = key

    def get_parent_bridge_id(self, comp_id: str) -> Any:
        """Retrieves the parent bridge ID for a given component ID."""
        return self.graph_funcs.graph_index[comp_id].get('parent_bridge_id', None)

    def assess_similarity(self, target_text: str, candidate_text: str) -> bool:
        """
        Uses an LLM to assess the semantic similarity between two disease descriptions.

        Args:
            target_text (str): The reference disease description.
            candidate_text (str): The candidate disease description to compare.

        Returns:
            bool: True if deemed similar, False otherwise.
        """
        prompt = f"""
        Target disease description: "{target_text}"
        Candidate disease description: "{candidate_text}"
        Based on the descriptions above, please determine if the candidate disease is similar to the target disease. 
        Focus on the core features of the target disease. Ignore aspects in the candidate that are not in the target. 
        Numerical values within a +/- 20% tolerance are acceptable.
        If they are similar, reply "yes", otherwise reply "no".
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm(messages)

            logger.info(f"Similarity Assessment Prompt:\n{prompt}")
            logger.info(f"Model Raw Response: {response}")
            logger.info(f"Model Response Content: {response.content}")

            judgment = response.content.strip().lower()
            logger.info(f"Processed Judgment: {judgment}")

            return any(keyword in judgment for keyword in ["yes", "是", "相似"])
        except Exception as e:
            logger.error(f"Error during similarity assessment: {e}")
            return False


### Utility Functions for String Processing ###

def split_checks(input_string):
    """Finds all occurrences of the pattern 'word[...]' in a string."""
    pattern = r'\w+\[.*?\]'
    result = re.findall(pattern, input_string)
    return result


def get_action_list(action: str) -> List[str]:
    """
    Extracts the first valid action string in the format 'action_type[args]'
    from the LLM's output, correctly handling nested brackets.

    Args:
        action (str): The raw action string from the LLM.

    Returns:
        List[str]: A list containing the first valid action, or an empty list.
    """
    pattern = re.compile(r'(\w+)\[')
    match = pattern.search(action)
    if not match:
        return []

    action_type = match.group(1)
    start = match.end()
    bracket_count = 1
    j = start
    while j < len(action):
        if action[j] == '[':
            bracket_count += 1
        elif action[j] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                break
        j += 1

    if bracket_count == 0:
        args = action[start:j]
        action_str = f"{action_type}[{args}]"
        return [action_str]
    else:
        # Mismatched brackets, ignore this action.
        return []


def remove_quotes(s: str) -> str:
    """Removes leading and trailing quotes (' or ") from a string."""
    symbols = ("'", '"', '\\')
    if s.startswith(symbols) and s.endswith(symbols):
        return s[1:-1]
    return s


def parse_action(s: str):
    """
    Parses a string in the format 'action_type[argument]' into a tuple.

    Args:
        s (str): The action string.

    Returns:
        A tuple (action_type, argument) or None if it doesn't match.
    """
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, s)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return None


def gpt_format_step(step: Any) -> str:
    """Extracts and cleans the content from an LLM response object."""
    return step.content.strip('\n').strip().replace('\n', '')


def normalize_answer(s: str) -> str:
    """Normalizes a string by lowercasing, removing punctuation, and articles."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer: str, key: str) -> bool:
    """Performs an Exact Match comparison between two normalized strings."""
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def split_args(argument: str) -> List[str]:
    """
    Intelligently splits an argument string by commas, respecting nested
    brackets and quotes.

    Args:
        argument (str): The raw argument string for an action.

    Returns:
        List[str]: A list of separated arguments.
    """
    args = []
    current_arg = ''
    bracket_level = 0
    in_quotes = False
    quote_char = ''
    i = 0
    while i < len(argument):
        c = argument[i]
        if c == '\\':
            i += 1
            continue

        if c in ['"', "'", '“', '”']:
            if not in_quotes:
                in_quotes = True
                quote_char = '"' if c in ['"', '“', '”'] else "'"
                current_arg += '"' if c in ['“', '”'] else "'"
            elif c == quote_char or (quote_char == '"' and c in ['“', '”']) or (quote_char == "'" and c in ["'"]):
                in_quotes = False
                current_arg += '"' if c in ['“', '”'] else "'"
            else:
                current_arg += c
        elif c in ['[', '(', '{']:
            if not in_quotes:
                bracket_level += 1
            current_arg += c
        elif c in [']', ')', '{']:
            if not in_quotes:
                bracket_level -= 1
            current_arg += c
        elif c == ',' and not in_quotes and bracket_level == 0:
            args.append(current_arg.strip())
            current_arg = ''
            if i + 1 < len(argument) and argument[i + 1] == ' ':
                i += 1
        else:
            current_arg += c
        i += 1
    if current_arg:
        args.append(current_arg.strip())
    return args


def standardize_quotes(s: str) -> str:
    """
    Replaces various non-standard quote characters in a string with
    standard English double or single quotes.

    Args:
        s (str): The original string.

    Returns:
        str: The string with standardized quotes.
    """
    return s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")