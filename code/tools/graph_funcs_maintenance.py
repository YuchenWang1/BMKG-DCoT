import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import ast
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher
from collections import deque


class graph_funcs():
    """
    A class to provide functionalities for interacting with a graph data structure,
    including node lookups, pathfinding, and information retrieval.
    """

    def __init__(self, graph):
        """
        Initializes the graph_funcs class with a graph object.

        Args:
            graph (dict): The graph data structure.
        """
        self._reset(graph)

    def _reset(self, graph):
        """
        Builds an index of all nodes in the graph for quick access.

        Args:
            graph (dict): The graph data structure.
        """
        graph_index = {}
        nid_set = set()
        for node_type in graph:
            for nid in graph[node_type]:
                assert nid not in nid_set
                nid_set.add(nid)
                node_data = graph[node_type][nid]
                node_data['node_type'] = node_type  # Add the node_type field.
                graph_index[nid] = node_data
        self.graph_index = graph_index

    def check_neighbours(self, node, neighbor_type=None):
        """
        Retrieves the neighbors of a given node.

        Args:
            node (str): The ID of the node.
            neighbor_type (str, optional): The type of neighbors to retrieve. Defaults to None.

        Returns:
            str: A string representation of the node's neighbors.
        """
        if neighbor_type:
            return str(self.graph_index[node]['neighbors'][neighbor_type])
        else:
            return str(self.graph_index[node]['neighbors'])

    def check_nodes(self, node, feature=None):
        """
        Retrieves the features of a given node.

        Args:
            node (str): The ID of the node.
            feature (str, optional): The specific feature to retrieve. Defaults to None.

        Returns:
            str: A string representation of the node's features.
        """
        if feature:
            return str(self.graph_index[node]['features'][feature])
        else:
            return str(self.graph_index[node]['features'])

    def check_degree(self, node, neighbor_type):
        """
        Calculates the degree of a node for a specific neighbor type.

        Args:
            node (str): The ID of the node.
            neighbor_type (str): The type of the neighbor connection.

        Returns:
            str: The degree of the node as a string.
        """
        return str(len(self.graph_index[node]['neighbors'][neighbor_type]))

    def get_node_attributes(self, node_id: str) -> Dict[str, str]:
        """
        Gets all features (attributes) of a specific node.

        Args:
            node_id (str): The ID of the node.

        Returns:
            Dict[str, str]: A dictionary of the node's attributes.
        """
        if node_id in self.graph_index:
            return self.graph_index[node_id].get('features', {})
        else:
            return {}

    def get_all_paths(self, start_id, max_depth=10):
        """
        Finds all possible paths starting from a node using DFS.

        Args:
            start_id (str): The starting node ID.
            max_depth (int, optional): The maximum depth for the search. Defaults to 10.

        Returns:
            List[List[str]]: A list of all found paths.
        """
        paths = []
        self._dfs(start_id, [start_id], paths, max_depth)
        return paths

    def _dfs(self, current, path, paths, max_depth):
        """
        Recursive helper function for Depth-First Search.

        Args:
            current (str): The current node ID in the traversal.
            path (List[str]): The path taken to reach the current node.
            paths (List[List[str]]): The list to store all found paths.
            max_depth (int): The maximum path length.
        """
        if len(path) > max_depth:
            return
        neighbors = self.graph_index.get(current, {}).get('neighbors', {})
        for neighbor_type, neighbor_ids in neighbors.items():
            for neighbor in neighbor_ids:
                if neighbor not in path:  # Avoid cycles.
                    new_path = path + [neighbor]
                    paths.append(new_path)
                    self._dfs(neighbor, new_path, paths, max_depth)

    def find_nodes_in_paths_with_similarity(self, start_id: str, query_name: str, similarity_threshold=0.65, max_depth=10):
        """
        Finds nodes in paths from a start node that match a query name,
        providing both exact and similar matches.

        Args:
            start_id (str): The starting node ID.
            query_name (str): The name to search for.
            similarity_threshold (float, optional): The threshold for a name to be considered similar. Defaults to 0.65.
            max_depth (int, optional): The maximum search depth. Defaults to 10.

        Returns:
            dict: A dictionary with 'exact' and 'similar' matches.
        """
        paths = self.get_all_paths(start_id, max_depth)
        exact_matches = set()
        similar_matches = set()

        for path in paths:
            for node in path:
                if 'name' in self.graph_index[node]['features']:
                    node_name = self.graph_index[node]['features']['name']
                    if node_name == query_name:
                        exact_matches.add((node_name, node))
                    else:
                        similarity = self.calculate_similarity(query_name, node_name)
                        if similarity >= similarity_threshold:
                            similar_matches.add((node_name, node))

        return {
            "exact": list(exact_matches),
            "similar": list(similar_matches)
        }

    def calculate_similarity(self, a: str, b: str) -> float:
        """
        Calculates the similarity ratio between two strings.

        Args:
            a (str): The first string.
            b (str): The second string.

        Returns:
            float: The similarity ratio between 0 and 1.
        """
        return SequenceMatcher(None, a, b).ratio()

    def get_nodes_by_type_in_paths(self, start_id, query_type, max_depth=10):
        """
        Finds all nodes of a specific type within paths starting from a given node.

        Args:
            start_id (str): The starting node ID.
            query_type (str): The node type to search for.
            max_depth (int, optional): The maximum search depth. Defaults to 10.

        Returns:
            List[str]: A list of node IDs of the specified type.
        """
        paths = self.get_all_paths(start_id, max_depth)
        nodes_of_type = set()
        query_type_all = f"{query_type}_nodes"
        for path in paths:
            for node in path:
                node_type = self.graph_index[node].get('node_type')
                if node_type == query_type_all:
                    nodes_of_type.add(node)
        return list(nodes_of_type)

    def get_all_nodes_by_type(self, node_type: str) -> List[str]:
        """
        Retrieves all node IDs of a specific type from the entire graph.

        Args:
            node_type (str): The node type to search for.

        Returns:
            List[str]: A list of all node IDs of that type.
        """
        result = []
        node_type_with_suffix = f"{node_type}_nodes"
        for nid, data in self.graph_index.items():
            if data['node_type'] == node_type_with_suffix:
                result.append(nid)
        return result

    def merge_node_info_bridge(self, node_id: str, upstream_depth=3) -> str:
        """
        Merges a bridge node's attributes into a single text string.

        Args:
            node_id (str): The ID of the bridge node.
            upstream_depth (int, optional): The depth to traverse for upstream info. Defaults to 3.

        Returns:
            str: A formatted string containing the node's information.
        """
        attributes = self.get_node_attributes(node_id)
        attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items() if k not in ['name', 'node_type']])
        return f"{attr_text}"

    def get_node_ids_by_name(self, name: str) -> List[str]:
        """
        Finds all node IDs that have an exact matching name.

        Args:
            name (str): The name to search for.

        Returns:
            List[str]: A list of matching node IDs.
        """
        return [node_id for node_id, data in self.graph_index.items()
                if data.get('features', {}).get('name') == name]

    def find_all_paths(self, start_id: str, end_id: str, max_depth: int = 10) -> List[List[str]]:
        """
        Finds all paths between a start node and an end node using an iterative approach.

        Args:
            start_id (str): The ID of the starting node.
            end_id (str): The ID of the ending node.
            max_depth (int, optional): The maximum path length. Defaults to 10.

        Returns:
            List[List[str]]: A list of all paths found between the two nodes.
        """
        paths = []
        stack = [(start_id, [start_id])]

        while stack:
            current, path = stack.pop()

            if len(path) > max_depth:
                continue

            neighbors = self.graph_index.get(current, {}).get('neighbors', {})
            if not neighbors:
                continue

            for rel, neighbor_list in neighbors.items():
                for neighbor in neighbor_list:
                    if neighbor == end_id:
                        paths.append(path + [neighbor])
                    elif neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        return paths

    def get_path_names(self, disease_name: str, original_bridge_name: str, max_depth: int = 10) -> List[List[str]]:
        """
        Finds all paths between a bridge and a disease and returns the names of nodes on those paths.

        Args:
            disease_name (str): The name of the disease node.
            original_bridge_name (str): The name of the bridge node.
            max_depth (int, optional): The maximum search depth. Defaults to 10.

        Returns:
            List[List[str]]: A list of paths, where each path is a list of node names.
        """
        bridge_ids = self.get_node_ids_by_name(original_bridge_name)
        if not bridge_ids:
            print(f"Bridge node with name '{original_bridge_name}' not found.")
            return []

        disease_ids = self.get_node_ids_by_name(disease_name)
        if not disease_ids:
            print(f"Disease node with name '{disease_name}' not found.")
            return []

        all_path_names = []
        for bridge_id in bridge_ids:
            for disease_id in disease_ids:
                paths = self.find_all_paths(bridge_id, disease_id, max_depth)
                for path in paths:
                    path_names = [self.graph_index[node_id]['features'].get('name', node_id) for node_id in path]
                    all_path_names.append(path_names)

        if not all_path_names:
            print(f"No path found from '{original_bridge_name}' to '{disease_name}'.")

        return all_path_names

    def merge_node_info(self, disease_id: str, bridge_id: str, max_depth: int = 4) -> str:
        """
        Merges information from a disease node and its connecting path from a bridge/component
        into a single descriptive string.

        Args:
            disease_id (str): The ID of the disease node.
            bridge_id (str): The ID of the starting bridge or component node.
            max_depth (int, optional): The maximum pathfinding depth. Defaults to 4.

        Returns:
            str: A formatted string summarizing the disease and its context.
        """
        if disease_id == bridge_id:
            attributes = self.get_node_attributes(disease_id)
            name_value = attributes.get('name', '')
            other_attrs = " ".join(
                [f"{k}:{v}" for k, v in attributes.items() if k not in ['name', 'node_type', 'unique_id']]
            )
            return f"{name_value} {other_attrs}".strip()

        attributes = self.get_node_attributes(disease_id)
        attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items() if k not in ['name', 'node_type', 'unique_id']])

        paths = self.find_all_paths(bridge_id, disease_id, max_depth)
        if not paths:
            return f"Path not found from component ID {bridge_id} to disease ID {disease_id}."

        selected_path = paths[0]
        path_info = []
        for node_id in selected_path[0:-1]:
            node_attributes = self.get_node_attributes(node_id)
            name = node_attributes.get('name', node_id)
            path_info.append(f"{name}")

        path_text = " ".join(path_info)
        return f"{path_text}, {attributes.get('name','')}, {attr_text} "

    def merge_node_info_attr(self, disease_id: str, bridge_id: str, max_depth: int = 4) -> str:
        """
        Merges information from a disease node and its path, focusing on attributes.

        Args:
            disease_id (str): The ID of the disease node.
            bridge_id (str): The ID of the starting bridge node.
            max_depth (int, optional): The maximum pathfinding depth. Defaults to 4.

        Returns:
            str: A formatted string of the disease's attributes.
        """
        attributes = self.get_node_attributes(disease_id)
        attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items() if k not in ['name', 'node_type', 'unique_id']])

        paths = self.find_all_paths(bridge_id, disease_id, max_depth)
        if not paths:
            return f"Path not found from bridge ID {bridge_id} to disease ID {disease_id}."

        selected_path = paths[0]
        path_info = []
        for node_id in selected_path[1:-1]:
            node_attributes = self.get_node_attributes(node_id)
            name = node_attributes.get('name', node_id)
            node_type = node_attributes.get('node_type', 'Unknown Type')
            path_info.append(f"{node_type}: {name}")

        path_text = ", ".join(path_info)
        return f"{attr_text} "

    def get_maintenance_measures(self, node_id: str) -> List[str]:
        """
        Retrieves the recommended maintenance measures for a given node (typically a disease node).

        Args:
            node_id (str): The ID of the node. Can be a string or a string representation of a list.

        Returns:
            List[str]: A list of maintenance measure names.
        """
        try:
            parsed = ast.literal_eval(node_id)
            if isinstance(parsed, list) and len(parsed) > 0:
                node_id = str(parsed[0])
        except Exception as e:
            pass

        measures = []
        node = self.graph_index.get(node_id, {})
        neighbors = node.get('neighbors', {})
        if "建议措施是" in neighbors: # "Recommended measure is"
            neighbor_value = neighbors["建议措施是"]
            if isinstance(neighbor_value, list):
                for mid in neighbor_value:
                    measure_name = self.graph_index.get(mid, {}).get('features', {}).get('name', 'Unknown Measure')
                    measures.append(measure_name)
            else:
                measure_name = self.graph_index.get(neighbor_value, {}).get('features', {}).get('name', 'Unknown Measure')
                measures.append(measure_name)
        return measures