import re
import os

def parse_metrics_from_file(file_name):
    metrics = {}

    # Regular expressions for each metric
    patterns = {
        "total_learning_time": r"Total learning time: ([\d.]+) seconds",
        "peak_memory_allocated": r"Peak memory allocated: (\d+)",
        "memory_allocated": r"Memory allocated: (\d+)",
        "memory_cached": r"Memory cached: (\d+)",
        "num_samples": r"num_samples:\s+(\d+)",
        "total_test_time": r"Total test time: ([\d.]+) seconds",
        "degree_mmd": r"Degree MMD:\s+([\d.]+)",
        "clustering_mmd": r"Clustering MMD:\s+([\d.]+)",
        "orbit_mmd": r"Orbit MMD\(4-node motif, 11 classes\):\s+([\d.]+)",
        "graph_kernel_mmd": r"Graph KernelMMD:\s+([\d.]+)",
        "Fréchet Distance": r"Fréchet Distance between set1 and set2:\s+([\d.]+)",
        "total_validation_time": r"Total validation time: ([\d.]+) seconds",
    }

    # Read log text from file
    try:
        with open(file_name, 'r') as file:
            log_text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return metrics

    # Extracting the last occurrence of each metric using regex
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text)
        if matches:
            last_value = matches[-1]
            metrics[key] = float(last_value) if '.' in last_value else int(last_value)

    return metrics

def parse_metrics_from_multiple_files(file_names):
    results = {}
    for file_name in file_names:
        if os.path.exists(file_name):
            print(f"Processing file: {file_name}")
            results[file_name] = parse_metrics_from_file(file_name)
        else:
            print(f"File not found: {file_name}")
    return results

def format_results_to_csv(results):
    csv_lines = []
    header = ["file_name"] + list(next(iter(results.values()), {}).keys())
    csv_lines.append(",".join(header))

    for file_name, metrics in results.items():
        line = [file_name] + [str(metrics.get(key, "")) for key in header[1:]]
        csv_lines.append(",".join(line))

    return "\n".join(csv_lines)

dir = "/home/tabei/prog/python/graphgen-cmdfs/results_label_new"

# Example usage
file_names = [
#    dir + "/res_citeseer_ego_label_temperature_0.8_new.txt",
#    dir + "/res_citeseer_ego_label_temperature_0.85_new.txt",
#    dir + "/res_citeseer_ego_label_temperature_0.9_new.txt",
#    dir + "/res_citeseer_ego_label_temperature_0.95_new.txt",
#    dir + "/res_citeseer_ego_label_temperature_1_new.txt",
#    dir + "/res_citeseer_community_label_temperature_0.8_new.txt",
#    dir + "/res_citeseer_community_label_temperature_0.85_new.txt",
#    dir + "/res_citeseer_community_label_temperature_0.9_new.txt",
#    dir + "/res_citeseer_community_label_temperature_0.95_new.txt",
#    dir + "/res_citeseer_community_label_temperature_1_new.txt",
#    dir + "/res_enzymes_label_temperature_0.8_new.txt",
#    dir + "/res_enzymes_label_temperature_0.85_new.txt",
#    dir + "/res_enzymes_label_temperature_0.9_new.txt",
#    dir + "/res_enzymes_label_temperature_0.95_new.txt",
#    dir + "/res_enzymes_label_temperature_1_new.txt",
    dir + "/res_proteins_label_temperature_0.8_new.txt",
    dir + "/res_proteins_label_temperature_0.85_new.txt",
    dir + "/res_proteins_label_temperature_0.9_new.txt",
    dir + "/res_proteins_label_temperature_0.95_new.txt",
    dir + "/res_proteins_label_temperature_1_new.txt",
]

parsed_results = parse_metrics_from_multiple_files(file_names)

# Display results as CSV
csv_output = format_results_to_csv(parsed_results)
print(csv_output)

# Display results
#for file_name, metrics in parsed_results.items():
#    print(f"\nResults for {file_name}:")
#    print(metrics)
