import networkx as nx
import numpy as np
import os
from urllib.request import urlretrieve
from typing import List
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

def load_citeseer_network() -> nx.Graph:
    """
    Load the Citeseer citation network from the raw data, removing self-loops
    """
    G = nx.Graph()
    
    with open("citeseer/citeseer.cites", 'r') as f:
        for line in f:
            source, target = line.strip().split('\t')
            # Only add edge if it's not a self-loop
            if source != target:
                G.add_edge(source, target)
    
    return G

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

def write_gspan(G: nx.Graph, file_path: str, graph_id: int):
    """
    Write a graph in gSpan format, ensuring no self-loops are written
    """
    # Create node mapping (original node ID -> sequential ID)
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    
    with open(file_path, 'a') as f:
        # Write graph header
        f.write(f't # {graph_id}\n')
        
        # Write nodes
        for node in G.nodes():
            f.write(f'v {node_map[node]} 0\n')
        
        # Write edges (excluding self-loops)
        for edge in G.edges():
            # Double-check that there are no self-loops
            if edge[0] != edge[1]:
                f.write(f'e {node_map[edge[0]]} {node_map[edge[1]]} 0\n')

def generate_ego_dataset(output_path: str, num_samples: int = 757) -> List[nx.Graph]:
    """
    Generate the ego network dataset from Citeseer and save in gSpan format
    """
    # Download and load Citeseer
    download_citeseer()
    citeseer_network = load_citeseer_network()
    
    # Create or clear the output file
    with open(output_path, 'w') as f:
        f.write('')  # Clear file contents
    
    ego_networks = []
    nodes = list(citeseer_network.nodes())
    np.random.shuffle(nodes)
    
    graph_id = 0
    for node in nodes:
        if len(ego_networks) >= num_samples:
            break
            
        ego_net = extract_ego_network(citeseer_network, node)
        n_nodes = ego_net.number_of_nodes()
        
        # Only keep networks within the specified size range (50 ≤ |V| ≤ 399)
        if 50 <= n_nodes <= 399:
            # Remove any remaining self-loops (should already be gone, but double-check)
            ego_net.remove_edges_from(nx.selfloop_edges(ego_net))
            
            ego_networks.append(ego_net)
            write_gspan(ego_net, output_path, graph_id)
            graph_id += 1
    
    print(f"Generated {len(ego_networks)} ego networks")
    print(f"Size range: {min(g.number_of_nodes() for g in ego_networks)} to "
          f"{max(g.number_of_nodes() for g in ego_networks)} nodes")
    
    return ego_networks

def main():
    output_path = "citeseer_ego.span"
    ego_networks = generate_ego_dataset(output_path)
    
    # Print statistics
    sizes = [g.number_of_nodes() for g in ego_networks]
    edges = [g.number_of_edges() for g in ego_networks]
    
    print(f"\nDataset statistics:")
    print(f"Number of networks: {len(ego_networks)}")
    print(f"Average network size: {np.mean(sizes):.2f} nodes")
    print(f"Average edge count: {np.mean(edges):.2f} edges")
    print(f"Median network size: {np.median(sizes):.2f} nodes")
    print(f"Size range: {min(sizes)} to {max(sizes)} nodes")
    print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    main()