import pandas as pd
import numpy as np

def read_txt_file(file_path):
    """Read a text file and return its content as a list."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def convert_to_gspan(dataset_name, output_file):
    """Convert dataset files to gspan format."""
    
    # Read all required files
    edges = pd.read_csv(f"{dataset_name}_A.txt", header=None, names=['source', 'target'])
    graph_indicators = read_txt_file(f"{dataset_name}_graph_indicator.txt")
    graph_labels = read_txt_file(f"{dataset_name}_graph_labels.txt")
    node_labels = read_txt_file(f"{dataset_name}_node_labels.txt")
    
    # Optional files
    try:
        edge_labels = read_txt_file(f"{dataset_name}_edge_labels.txt")
        has_edge_labels = True
    except FileNotFoundError:
        has_edge_labels = False
    
    # Create a mapping of node_id to graph_id
    node_to_graph = {i+1: int(graph_id) for i, graph_id in enumerate(graph_indicators)}
    
    # Group edges by graph
    edges['graph_id'] = edges['source'].map(node_to_graph)
    grouped_edges = edges.groupby('graph_id')
    
    # Open output file
    with open(output_file, 'w') as f:
        # Process each graph
        for graph_id in range(1, len(graph_labels) + 1):
            # Write graph header
            f.write(f"t # {graph_id-1}\n")
            
            # Get nodes for this graph
            graph_nodes = [i for i, g in node_to_graph.items() if g == graph_id]
            
            # Write nodes
            for node_id in graph_nodes:
                label = node_labels[node_id-1]
                f.write(f"v {node_id-1} {label}\n")
            
            # Write edges
            if graph_id in grouped_edges.groups:
                graph_edges = grouped_edges.get_group(graph_id)
                for _, edge in graph_edges.iterrows():
                    source = edge['source'] - 1  # Convert to 0-based indexing
                    target = edge['target'] - 1
                    if has_edge_labels:
                        edge_label = edge_labels[_]
                        f.write(f"e {source} {target} {edge_label}\n")
                    else:
                        f.write(f"e {source} {target}\n")

def main():
    # Example usage
    dataset_name = "/home/tabei/prog/python/graph-generation/dataset/ENZYMES/ENZYMES"
    output_file = "/home/tabei/prog/python/graph-generation/dataset/ENZYMES/ENZYMES.gspan"
    convert_to_gspan(dataset_name, output_file)

if __name__ == "__main__":
    main()