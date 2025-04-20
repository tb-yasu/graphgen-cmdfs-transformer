import networkx as nx
import numpy as np
import random
from typing import List

def generate_er_graph_no_selfloops(n: int, p: float) -> nx.Graph:
    """
    Generate Erdős-Rényi graph without self-loops
    
    Args:
        n: Number of nodes
        p: Probability of edge creation
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):  # Only consider upper triangle to avoid self-loops
            if random.random() < p:
                G.add_edge(i, j)
    
    return G

def generate_community_graph(min_nodes: int = 60, max_nodes: int = 160, p_intra: float = 0.3) -> nx.Graph:
    """
    Generate a two-community graph according to the paper specifications.
    
    Args:
        min_nodes: Minimum total number of nodes
        max_nodes: Maximum total number of nodes
        p_intra: Probability of edge creation within communities
    """
    # Randomly select total number of nodes (must be even to split equally)
    total_nodes = random.randrange(min_nodes // 2, max_nodes // 2 + 1) * 2
    community_size = total_nodes // 2
    
    # Generate two E-R graphs for the communities
    g1 = generate_er_graph_no_selfloops(community_size, p_intra)
    g2 = generate_er_graph_no_selfloops(community_size, p_intra)
    
    # Relabel nodes of second community to avoid overlap
    mapping = {node: node + community_size for node in g2.nodes()}
    g2 = nx.relabel_nodes(g2, mapping)
    
    # Combine the communities
    G = nx.Graph()
    G.add_edges_from(g1.edges())
    G.add_edges_from(g2.edges())
    
    # Add inter-community edges (0.05|V| edges as per paper)
    num_inter_edges = int(0.05 * total_nodes)
    nodes1 = list(range(community_size))
    nodes2 = list(range(community_size, total_nodes))
    
    added_edges = 0
    while added_edges < num_inter_edges:
        v1 = random.choice(nodes1)
        v2 = random.choice(nodes2)
        if not G.has_edge(v1, v2):
            G.add_edge(v1, v2)
            added_edges += 1
    
    return G

def write_gspan(G: nx.Graph, file_path: str, graph_id: int):
    """
    Write a graph in gSpan format
    """
    with open(file_path, 'a') as f:
        # Write graph header
        f.write(f't # {graph_id}\n')
        
        # Write nodes
        # Nodes in first community get label 0, nodes in second community get label 1
        n = len(G.nodes())
        for node in G.nodes():
            label = 0 if node < n//2 else 1
            f.write(f'v {node} {label}\n')
        
        # Write edges
        for edge in G.edges():
            f.write(f'e {edge[0]} {edge[1]} 0\n')

def generate_dataset(num_graphs: int = 500, output_path: str = "community.span") -> List[nx.Graph]:
    """
    Generate the complete community graph dataset and save in gSpan format
    """
    # Clear output file
    with open(output_path, 'w') as f:
        f.write('')
    
    graphs = []
    for i in range(num_graphs):
        G = generate_community_graph()
        graphs.append(G)
        write_gspan(G, output_path, i)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} graphs")
    
    return graphs

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate dataset
    graphs = generate_dataset()
    
    # Print statistics
    sizes = [g.number_of_nodes() for g in graphs]
    edges = [g.number_of_edges() for g in graphs]
    intra_edges = []
    inter_edges = []
    
    for g in graphs:
        n = len(g.nodes())
        inter = sum(1 for (u, v) in g.edges() 
                   if (u < n//2 and v >= n//2) or (u >= n//2 and v < n//2))
        intra_edges.append(g.number_of_edges() - inter)
        inter_edges.append(inter)
    
    print("\nDataset statistics:")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Node count - Mean: {np.mean(sizes):.1f}, Min: {min(sizes)}, Max: {max(sizes)}")
    print(f"Edge count - Mean: {np.mean(edges):.1f}, Min: {min(edges)}, Max: {max(edges)}")
    print(f"Intra-community edges - Mean: {np.mean(intra_edges):.1f}")
    print(f"Inter-community edges - Mean: {np.mean(inter_edges):.1f}")
    
    # Verify no self-loops exist
    total_self_loops = sum(len(list(nx.selfloop_edges(g))) for g in graphs)
    print(f"Total self-loops found: {total_self_loops} (should be 0)")

if __name__ == "__main__":
    main()

