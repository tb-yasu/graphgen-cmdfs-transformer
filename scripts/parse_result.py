import re
import matplotlib.pyplot as plt

def parse_file(file_path):
    # Initialize variables to store the extracted data
    max_total_learning_time = 0
    max_peak_memory_allocated = 0
    max_total_validation_time = 0

    epoch_analysis1 = []
    epoch_analysis = []


    epoch = 0
    novelty = 0
    uniqueness = 0
    diversity = 0
    average_nodes = 0
    std_nodes = 0
    average_edges = 0
    std_edges = 0
    average_clustering_coefficient = 0
    std_clustering_coefficient = 0
    average_assortativity = 0
    std_assortativity = 0
    degree_mmd = 0
    clustering_mmd = 0
    orbit_mdd = 0
    # Read the entire file content
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.match(r"Epoch: (\d+),", line):
            epoch = int(re.search(r"Epoch: (\d+),", line).group(1))
          
        # Extract total learning time
        if "Total learning time:" in line:
            learning_time = float(re.search(r"Total learning time: ([\d.]+)", line).group(1))
            max_total_learning_time = max(max_total_learning_time, learning_time)

        # Extract peak memory allocated
        if "Peak memory allocated:" in line:
            peak_memory = int(re.search(r"Peak memory allocated: (\d+)", line).group(1))
            max_peak_memory_allocated = max(max_peak_memory_allocated, peak_memory)

        # Extract total validation time
        if "Total validation time:" in line:
            validation_time = float(re.search(r"Total validation time: ([\d.]+)", line).group(1))
            max_total_validation_time = max(max_total_validation_time, validation_time)

        if "Novelty" in line:
            novelty = re.search(r"Novelty: ([\d.]+)%", line)
          
        if "Uniqueness" in line:
            uniqueness = re.search(r"Uniqueness: ([\d.]+)%", line)

        if "Diversity" in line:
            diversity = re.search(r"Diversity: ([\d.]+)%", line)

            metrics = {
                "epoch": epoch,
                "novelty": float(novelty.group(1)) if novelty else None,
                "uniqueness": float(uniqueness.group(1)) if uniqueness else None,
                "diversity": float(diversity.group(1)) if diversity else None,
            }
            epoch_analysis1.append(metrics)
        if "平均ノード数" in line:
            average_nodes = line.split(": ")[1].split(" ± ")[0].strip()
            std_nodes = line.split(": ")[1].split(" ± ")[1].strip()
            average_nodes = float(average_nodes)
            std_nodes = float(std_nodes)

        if "平均エッジ数" in line:
            average_edges = line.split(": ")[1].split(" ± ")[0].strip()
            std_edges = line.split(": ")[1].split(" ± ")[1].strip()
            average_edges = float(average_edges)
            std_edges = float(std_edges)
        
        if "平均クラスタリング係数" in line:
            average_clustering_coefficient = line.split(": ")[1].split(" ± ")[0].strip()
            std_clustering_coefficient = line.split(": ")[1].split(" ± ")[1].strip()
            average_clustering_coefficient = float(average_clustering_coefficient)
            std_clustering_coefficient = float(std_clustering_coefficient)
        
        if "平均Assortativity" in line:
            average_assortativity = line.split(": ")[1].split(" ± ")[0].strip()
            std_assortativity = line.split(": ")[1].split(" ± ")[1].strip()
            average_assortativity = float(average_assortativity)
            std_assortativity = float(std_assortativity)

        if "degree_mmd" in line:
            degree_mmd = line.split(": ")[1].strip()
            degree_mmd = float(degree_mmd)

        if "clustering_mmd" in line:
            clustering_mmd = line.split(": ")[1].strip()
            clustering_mmd = float(clustering_mmd)

        if "orbit_mdd:" in line:
            orbit_mdd = line.split(": ")[1].strip()
            orbit_mdd = float(orbit_mdd)
        
        if "Total validation time" in line:
            total_validation_time = line.split(": ")[1].strip()
            total_validation_time = total_validation_time.split(" ")[0]
            total_validation_time = float(total_validation_time)

            metrics.update({
                "average_nodes": average_nodes,
                "std_nodes": std_nodes,
                "average_edges": average_edges,
                "std_edges": std_edges,
                "average_clustering_coefficient": average_clustering_coefficient,
                "std_clustering_coefficient": std_clustering_coefficient,
                "average_assortativity": average_assortativity,
                "std_assortativity": std_assortativity,
                "degree_mmd": degree_mmd,
                "clustering_mmd": clustering_mmd,
                "orbit_mdd": orbit_mdd,
                "total_validation_time": total_validation_time,
            })

            epoch_analysis.append(metrics)
#            print("Epoch Analysis:")
#            for analysis in epoch_analysis:
#                print(analysis)

    # Print results
    print("Max Total Learning Time:", max_total_learning_time)
    print("Max Peak Memory Allocated:", max_peak_memory_allocated)
    print("Max Total Validation Time:", max_total_validation_time)
#    print("Epoch Analysis:")
#    for analysis in epoch_analysis:
#        print(analysis)

    # Plot results
    plot_graph1(epoch_analysis1)
    plot_graphs(epoch_analysis)


def plot_graph1(epoch_analysis1):
    epochs = [entry["epoch"] for entry in epoch_analysis1]
    novelty = [entry.get("novelty", None) for entry in epoch_analysis1]
    uniqueness = [entry.get("uniqueness", None) for entry in epoch_analysis1]
    diversity = [entry.get("diversity", None) for entry in epoch_analysis1]

    plt.figure()
    plt.plot(epochs, novelty, label="Novelty")
    plt.plot(epochs, uniqueness, label="Uniqueness")
    plt.plot(epochs, diversity, label="Diversity")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Metrics vs Epoch")
    plt.legend()
    plt.savefig("metrics_vs_epoch.png")

def plot_graphs(epoch_analysis):
    epochs = [entry["epoch"] for entry in epoch_analysis]
    average_nodes = [entry.get("average_nodes", None) for entry in epoch_analysis]
    average_edges = [entry.get("average_edges", None) for entry in epoch_analysis]
    average_clustering_coefficient = [entry.get("average_clustering_coefficient", None) for entry in epoch_analysis]
    average_assortativity = [entry.get("average_assortativity", None) for entry in epoch_analysis]
    degree_mmd = [entry.get("degree_mmd", None) for entry in epoch_analysis]
    clustering_mmd = [entry.get("clustering_mmd", None) for entry in epoch_analysis]
    orbit_mdd = [entry.get("orbit_mdd", None) for entry in epoch_analysis]
    total_validation_time = [entry.get("total_validation_time", None) for entry in epoch_analysis]

    # Plot average nodes
    plt.figure()
    plt.plot(epochs, average_nodes, label="Average Nodes")
    plt.xlabel("Epoch")
    plt.ylabel("Average Nodes")
    plt.title("Average Nodes vs Epoch")
    plt.legend()
    plt.savefig("average_nodes_vs_epoch.png")

    # Plot average edges
    plt.figure()
    plt.plot(epochs, average_edges, label="Average Edges")
    plt.xlabel("Epoch")
    plt.ylabel("Average Edges")
    plt.title("Average Edges vs Epoch")
    plt.legend()
    plt.savefig("average_edges_vs_epoch.png")

    # Plot average clustering coefficient
    plt.figure()
    plt.plot(epochs, average_clustering_coefficient, label="Average Clustering Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Average Clustering Coefficient")
    plt.title("Average Clustering Coefficient vs Epoch")
    plt.legend()
    plt.savefig("average_clustering_coefficient_vs_epoch.png")

    # Plot average assortativity
    plt.figure()
    plt.plot(epochs, average_assortativity, label="Average Assortativity")
    plt.xlabel("Epoch")
    plt.ylabel("Average Assortativity")
    plt.title("Average Assortativity vs Epoch")
    plt.legend()
    plt.savefig("average_assortativity_vs_epoch.png")


    # Plot degree mmd
    plt.figure()
    plt.plot(epochs, degree_mmd, label="Degree MMD")
    plt.xlabel("Epoch")
    plt.ylabel("Degree MMD")
    plt.title("Degree MMD vs Epoch")
    plt.legend()
    plt.savefig("degree_mmd_vs_epoch.png")

    # Plot clustering mmd
    plt.figure()
    plt.plot(epochs, clustering_mmd, label="Clustering MMD")
    plt.xlabel("Epoch")
    plt.ylabel("Clustering MMD")
    plt.title("Clustering MMD vs Epoch")
    plt.legend()
    plt.savefig("clustering_mmd_vs_epoch.png")

    # Plot orbit mdd
    plt.figure()
    plt.plot(epochs, orbit_mdd, label="Orbit MDD")
    plt.xlabel("Epoch")
    plt.ylabel("Orbit MDD")
    plt.title("Orbit MDD vs Epoch")
    plt.legend()
    plt.savefig("orbit_mdd_vs_epoch.png")

    # Plot total validation time
    plt.figure()
    plt.plot(epochs, total_validation_time, label="Total Validation Time")
    plt.xlabel("Epoch")
    plt.ylabel("Total Validation Time")
    plt.title("Total Validation Time vs Epoch")
    plt.legend()
    plt.savefig("total_validation_time_vs_epoch.png")

    # 各項目の最後に格納された値のみ出力
    print("Final Average Nodes: {:.2f}".format(average_nodes[-1]))
    print("Final Average Edges: {:.2f}".format(average_edges[-1]))
    print("Final Average Clustering Coefficient: {:.2f}".format(average_clustering_coefficient[-1]))
    print("Final Average Assortativity: {:.2f}".format(average_assortativity[-1]))
    print("Final Degree MMD: {:.2f}".format(degree_mmd[-1]))
    print("Final Clustering MMD: {:.2f}".format(clustering_mmd[-1]))
    print("Final Orbit MDD: {:.3f}".format(orbit_mdd[-1]))
    print("Final Total Validation Time: {:.2f}".format(total_validation_time[-1]))

# Example usage
# Provide the path to your file
data_file_path = "res_ego_small_0.8.txt"
data_file_path = "res_community_0.8.txt"
data_file_path = "res_chem_0.8.txt"
parse_file(data_file_path)
