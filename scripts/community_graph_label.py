import networkx as nx
import numpy as np
import random
from typing import List, Dict, Tuple
import argparse

def generate_er_graph_no_selfloops(n: int, p: float, 
                                 num_node_labels: int = 1,
                                 num_edge_labels: int = 1) -> Tuple[nx.Graph, Dict, Dict]:
    """
    Generate Erdős-Rényi graph without self-loops, with node and edge labels
    
    Args:
        n: Number of nodes
        p: Probability of edge creation
        num_node_labels: Number of different node labels
        num_edge_labels: Number of different edge labels
    
    Returns:
        Tuple of (Graph, node_labels, edge_labels)
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Generate random node labels
    node_labels = {i: random.randrange(num_node_labels) for i in range(n)}
    
    # Generate edges with random labels
    edge_labels = {}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)
                edge_labels[(i, j)] = random.randrange(num_edge_labels)
                edge_labels[(j, i)] = edge_labels[(i, j)]  # 無向グラフなので同じラベル
    
    return G, node_labels, edge_labels

def generate_community_graph(min_nodes: int = 60, 
                           max_nodes: int = 160, 
                           p_intra: float = 0.3,
                           num_node_labels: int = 1,
                           num_edge_labels: int = 1) -> Tuple[nx.Graph, Dict, Dict]:
    """
    Generate a two-community graph with labels
    """
    total_nodes = random.randrange(min_nodes // 2, max_nodes // 2 + 1) * 2
    community_size = total_nodes // 2
    
    # Generate two communities
    g1, node_labels1, edge_labels1 = generate_er_graph_no_selfloops(
        community_size, p_intra, num_node_labels, num_edge_labels)
    g2, node_labels2, edge_labels2 = generate_er_graph_no_selfloops(
        community_size, p_intra, num_node_labels, num_edge_labels)
    
    # Relabel second community nodes
    mapping = {node: node + community_size for node in g2.nodes()}
    g2 = nx.relabel_nodes(g2, mapping)
    
    # Update labels for second community
    node_labels2 = {mapping[k]: v for k, v in node_labels2.items()}
    edge_labels2 = {(mapping[u], mapping[v]): l for (u, v), l in edge_labels2.items()}
    
    # Combine the communities
    G = nx.Graph()
    G.add_edges_from(g1.edges())
    G.add_edges_from(g2.edges())
    
    # Combine labels
    node_labels = {**node_labels1, **node_labels2}
    edge_labels = {**edge_labels1, **edge_labels2}
    
    # Add inter-community edges with random labels
    num_inter_edges = int(0.05 * total_nodes)
    nodes1 = list(range(community_size))
    nodes2 = list(range(community_size, total_nodes))
    
    added_edges = 0
    while added_edges < num_inter_edges:
        v1 = random.choice(nodes1)
        v2 = random.choice(nodes2)
        if not G.has_edge(v1, v2):
            G.add_edge(v1, v2)
            label = random.randrange(num_edge_labels)
            edge_labels[(v1, v2)] = label
            edge_labels[(v2, v1)] = label
            added_edges += 1
    
    return G, node_labels, edge_labels

def write_gspan(G: nx.Graph, 
                node_labels: Dict[int, int], 
                edge_labels: Dict[Tuple[int, int], int],
                file_path: str, 
                graph_id: int,
                community_labels: bool = True):
    """
    Write a labeled graph in gSpan format
    
    Args:
        community_labels: If True, add community membership as additional node label
    """
    with open(file_path, 'a') as f:
        f.write(f't # {graph_id}\n')
        
        # Write nodes
        n = len(G.nodes())
        for node in G.nodes():
            # Combine community membership with node label if requested
            if community_labels:
                comm_label = 0 if node < n//2 else 1
                label = f"{comm_label}_{node_labels[node]}"  # コミュニティとノードラベルを組み合わせる
            else:
                label = str(node_labels[node])
            f.write(f'v {node} {label}\n')
        
        # Write edges
        for edge in G.edges():
            label = edge_labels[edge]
            f.write(f'e {edge[0]} {edge[1]} {label}\n')

def generate_dataset(num_graphs: int = 500, 
                    output_path: str = "community.span",
                    num_node_labels: int = 1,
                    num_edge_labels: int = 1,
                    community_labels: bool = True) -> List[Tuple[nx.Graph, Dict, Dict]]:
    """
    Generate the complete labeled community graph dataset
    """
    with open(output_path, 'w') as f:
        f.write('')
    
    graphs = []
    for i in range(num_graphs):
        G, node_labels, edge_labels = generate_community_graph(
            num_node_labels=num_node_labels,
            num_edge_labels=num_edge_labels
        )
        graphs.append((G, node_labels, edge_labels))
        write_gspan(G, node_labels, edge_labels, output_path, i, community_labels)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} graphs")
    
    return graphs

def main():
    parser = argparse.ArgumentParser(description='Generate labeled community graphs dataset')
    parser.add_argument('--num-graphs', type=int, default=500,
                      help='Number of graphs to generate')
    parser.add_argument('--num-node-labels', type=int, default=3,
                      help='Number of node labels (excluding community labels)')
    parser.add_argument('--num-edge-labels', type=int, default=3,
                      help='Number of edge labels')
    parser.add_argument('--output', type=str, default='community_label.span',
                      help='Output file path')
    parser.add_argument('--no-community-labels', action='store_true',
                      help='Do not include community membership in node labels')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate dataset
    graphs = generate_dataset(
        num_graphs=args.num_graphs,
        output_path=args.output,
        num_node_labels=args.num_node_labels,
        num_edge_labels=args.num_edge_labels,
        community_labels=not args.no_community_labels
    )
    
    # Print statistics
    total_nodes = sum(len(g[0].nodes()) for g in graphs)
    total_edges = sum(len(g[0].edges()) for g in graphs)
    node_label_counts = {}
    edge_label_counts = {}
    
    for _, node_labels, edge_labels in graphs:
        for label in node_labels.values():
            node_label_counts[label] = node_label_counts.get(label, 0) + 1
        for label in edge_labels.values():
            edge_label_counts[label] = edge_label_counts.get(label, 0) + 1
    
    print("\nDataset statistics:")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Average nodes per graph: {total_nodes/len(graphs):.1f}")
    print(f"Average edges per graph: {total_edges/len(graphs):.1f}")
    print(f"\nNode label distribution:")
    for label, count in sorted(node_label_counts.items()):
        print(f"Label {label}: {count/total_nodes:.3f}")
    print(f"\nEdge label distribution:")
    for label, count in sorted(edge_label_counts.items()):
        print(f"Label {label}: {count/total_edges:.3f}")

if __name__ == "__main__":
    main()