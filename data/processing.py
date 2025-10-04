"""
该脚本将 Neo4j 导出的图谱 JSON 转换为便于检索的 graph.json。
Convert a Neo4j-exported graph JSON into a searchable `graph.json`.
"""

import os
import json
from tqdm import tqdm
from collections import defaultdict

def extract_unique_id(elementId):
    """
    Extract the unique identifier from an elementId (the part after the last colon).
    """
    return elementId.split(':')[-1]

def convert_input_to_graph(input_file, output_file):
    """
    Convert the raw Neo4j-exported JSON into the target `graph.json` structure.

    Args:
        input_file (str): Path to the raw Neo4j-exported graph JSON file.
        output_file (str): Path to the output `graph.json` file.
    """
    # Initialize the graph container, grouped by node type.
    graph = defaultdict(dict)

    # Track node IDs that have been added to avoid duplicates.
    id_set = set()

    # Map elementId to unique_id.
    elementId_to_unique_id = {}

    # Map unique_id to node_type.
    unique_id_to_node_type = {}

    # Read the input JSON file.
    # Try to parse the entire file as a JSON array.
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            data = []
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Iterate over each relationship entry.
    for entry in tqdm(data, desc="Processing"):
        n = entry.get('n', {})
        m = entry.get('m', {})
        r = entry.get('r', {})

        # Process source node `n`.
        n_elementId = n.get('elementId')
        n_unique_id = extract_unique_id(n_elementId) if n_elementId else None
        n_labels = n.get('labels', [])
        n_name = n.get('properties', {}).get('name', '')
        n_features = n.get('properties', {}).copy() if n.get('properties') else {}
        n_features['name'] = n_name
        n_features['node_type'] = n_labels[0] if n_labels else 'Unknown'

        # Record the mapping from elementId to unique_id.
        if n_elementId:
            elementId_to_unique_id[n_elementId] = n_unique_id

        # If the node has not been added, append it to the graph.
        if n_unique_id and n_unique_id not in id_set:
            id_set.add(n_unique_id)
            node_type = n_features['node_type']
            graph[node_type][n_unique_id] = {
                'features': n_features,
                'neighbors': defaultdict(list)
            }
            unique_id_to_node_type[n_unique_id] = node_type

        # Process target node `m`.
        m_elementId = m.get('elementId')
        m_unique_id = extract_unique_id(m_elementId) if m_elementId else None
        m_labels = m.get('labels', [])
        m_name = m.get('properties', {}).get('name', '')
        m_features = m.get('properties', {}).copy()
        m_features['name'] = m_name
        m_features['node_type'] = m_labels[0] if m_labels else 'Unknown'

        # Record the mapping from elementId to unique_id.
        if m_elementId:
            elementId_to_unique_id[m_elementId] = m_unique_id

        # If the node has not been added, append it to the graph.
        if m_unique_id and m_unique_id not in id_set:
            id_set.add(m_unique_id)
            node_type = m_features['node_type']
            graph[node_type][m_unique_id] = {
                'features': m_features,
                'neighbors': defaultdict(list)
            }
            unique_id_to_node_type[m_unique_id] = node_type

        # Process relationship `r`.
        r_elementId = r.get('elementId')
        r_type = r.get('type')
        r_properties = r.get('properties', {})
        r_start = r.get('startNodeElementId')
        r_end = r.get('endNodeElementId')

        # Resolve unique_id by elementId.
        start_unique_id = elementId_to_unique_id.get(r_start)
        end_unique_id = elementId_to_unique_id.get(r_end)

        if not start_unique_id or not end_unique_id:
            # Skip the relationship if corresponding unique_id cannot be found.
            continue

        # Get the source node's type.
        source_type = unique_id_to_node_type.get(start_unique_id)

        if source_type and start_unique_id in graph[source_type]:
            # Add the target node into the source node's `neighbors`.
            graph[source_type][start_unique_id]['neighbors'][r_type].append({
                'target_id': end_unique_id,
                'edge_properties': r_properties
            })

    # Build the final graph dict that groups nodes by type.
    graph_dict = {'node_types': []}

    for node_type, nodes in graph.items():
        graph_dict['node_types'].append(node_type)
        graph_dict[f"{node_type}_nodes"] = {}

        for node_id, node_data in nodes.items():

            node_data['neighbors'] = dict(node_data['neighbors'])
            graph_dict[f"{node_type}_nodes"][node_id] = node_data

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(graph_dict, f_out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Input file path.
    input_file = 'raw.json'
    # Output file path.
    output_file = 'graph.json'
    # Run the conversion.
    convert_input_to_graph(input_file, output_file)