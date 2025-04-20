import networkx as nx
import numpy as np
import os
from urllib.request import urlretrieve
from typing import List, Dict, Tuple
import tarfile

def download_citeseer():
    """
    Download the Citeseer dataset
    """
    base_url = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    target_path = "citeseer.tgz"
    
    if not os.path.exists(target_path):
        print("Downloading Citeseer dataset...")
        urlretrieve(base_url, target_path)
    
    if not os.path.exists("citeseer"):
        print("Extracting dataset...")
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall()

def load_node_labels() -> Dict[str, int]:
    """
    Load node labels from Citeseer dataset
    Categories: Agents, AI, DB, IR, ML, HCI
    """
    node_labels = {}
    label_map = {
        'Agents': 0,
        'AI': 1,
        'DB': 2,
        'IR': 3,
        'ML': 4,
        'HCI': 5
    }
    
    with open("citeseer/citeseer.content", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = parts[0]
            label = parts[-1]
            node_labels[node_id] = label_map[label]
    
    return node_labels

def determine_edge_label(source_label: int, target_label: int) -> int:
    """
    Determine edge label based on the categories of source and target nodes
    
    Edge label rules:
    0: Same category citation
    1: Cross-category within related fields (e.g., AI-ML, DB-IR)
    2: Cross-category between distant fields
    """
    if source_label == target_label:
        return 0
    
    # Define related fields
    related_fields = {
        frozenset([0, 1]),  # Agents-AI
        frozenset([1, 4]),  # AI-ML
        frozenset([2, 3]),  # DB-IR
        frozenset([3, 4]),  # IR-ML
        frozenset([0, 5])   # Agents-HCI
    }
    
    pair = frozenset([source_label, target_label])
    return 1 if pair in related_fields else 2

def load_citeseer_network(node_labels: Dict[str, int]) -> Tuple[nx.Graph, Dict[Tuple[str, str], int]]:
    """
    Load the Citeseer citation network with node and edge labels
    """
    G = nx.Graph()
    edge_labels = {}
    
    # Add nodes with labels
    for node, label in node_labels.items():
        G.add_node(node, label=label)
    
    # Add edges with labels
    with open("citeseer/citeseer.cites", 'r') as f:
        for line in f:
            source, target = line.strip().split('\t')
            if source != target and source in node_labels and target in node_labels:
                source_label = node_labels[source]
                target_label = node_labels[target]
                edge_label = determine_edge_label(source_label, target_label)
                G.add_edge(source, target)
                edge_labels[(source, target)] = edge_label
                edge_labels[(target, source)] = edge_label  # For undirected graph
    
    return G, edge_labels

def extract_ego_network(G: nx.Graph, node: str, radius: int = 3) -> nx.Graph:
    """
    Extract an ego network for a given node with specified radius
    """
    ego_nodes = {node}
    current_boundary = {node}
    
    for _ in range(radius):
        next_boundary = set()
        for n in current_boundary:
            neighbors = set(G.neighbors(n))
            next_boundary.update(neighbors - ego_nodes)
        ego_nodes.update(next_boundary)
        current_boundary = next_boundary
    
    return G.subgraph(ego_nodes).copy()

def write_gspan(G: nx.Graph, edge_labels: Dict[Tuple[str, str], int], 
                file_path: str, graph_id: int):
    """
    Write a labeled graph in gSpan format
    """
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    
    with open(file_path, 'a') as f:
        f.write(f't # {graph_id}\n')
        
        # Write nodes with their category labels
        for node in G.nodes():
            label = G.nodes[node]['label']
            f.write(f'v {node_map[node]} {label}\n')
        
        # Write edges with their relationship labels
        for edge in G.edges():
            if edge[0] != edge[1]:
                # Get edge label from the dictionary
                label = edge_labels.get((edge[0], edge[1]), 0)
                f.write(f'e {node_map[edge[0]]} {node_map[edge[1]]} {label}\n')

def generate_ego_dataset(output_path: str, num_samples: int = 757) -> List[nx.Graph]:
    """
    Generate the labeled ego network dataset from Citeseer
    """
    download_citeseer()
    node_labels = load_node_labels()
    citeseer_network, edge_labels = load_citeseer_network(node_labels)
    
    with open(output_path, 'w') as f:
        f.write('# Citeseer Ego Networks with Labels\n')
        f.write('# Node labels: 0=Agents, 1=AI, 2=DB, 3=IR, 4=ML, 5=HCI\n')
        f.write('# Edge labels: 0=Same category, 1=Related fields, 2=Distant fields\n\n')
    
    ego_networks = []
    nodes = list(citeseer_network.nodes())
    np.random.shuffle(nodes)
    
    graph_id = 0
    for node in nodes:
        if len(ego_networks) >= num_samples:
            break
            
        ego_net = extract_ego_network(citeseer_network, node)
        n_nodes = ego_net.number_of_nodes()
        
        if 50 <= n_nodes <= 399:
            ego_networks.append(ego_net)
            write_gspan(ego_net, edge_labels, output_path, graph_id)
            graph_id += 1
    
    return ego_networks

def main():
    output_path = "citeseer_ego_labeled.span"
    ego_networks = generate_ego_dataset(output_path)
    
    # Compute and print statistics
    sizes = [g.number_of_nodes() for g in ego_networks]
    edges = [g.number_of_edges() for g in ego_networks]
    
    node_label_counts = {i: 0 for i in range(6)}
    for g in ego_networks:
        for node in g.nodes():
            label = g.nodes[node]['label']
            node_label_counts[label] += 1
    
    print(f"\nDataset statistics:")
    print(f"Number of networks: {len(ego_networks)}")
    print(f"Average network size: {np.mean(sizes):.2f} nodes")
    print(f"Average edge count: {np.mean(edges):.2f} edges")
    print(f"\nNode label distribution:")
    categories = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    for i, cat in enumerate(categories):
        print(f"{cat}: {node_label_counts[i]} nodes")
    
    print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    main()