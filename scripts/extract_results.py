import os
import re
import csv

# フォルダ内のファイル名をリストアップ
#files = [
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_128_temperature_0.85_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_128_temperature_0.8_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_128_temperature_0.95_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_128_temperature_0.9_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_128_temperature_1_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_64_temperature_0.85_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_64_temperature_0.8_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_64_temperature_0.95_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_64_temperature_0.9_new.txt",
#    "./results_nolabel/res_citeseer_community_nolabel_hiddensize_64_temperature_1_new.txt",
#]

#files = [
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_128_templerature_0.85.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_128_templerature_0.8.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_128_templerature_0.95.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_128_templerature_0.9.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_128_templerature_1.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_64_templerature_0.85.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_64_templerature_0.8.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_64_templerature_0.95.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_64_templerature_0.9.txt",
#    "./results_nolabel/res_citeseer_ego_nolabel_hiddensize_64_templerature_1.txt",
#]

#files = [
#    "./results_nolabel/res_enzymes_hiddensize_128_temperature_0.85.txt",
#    "./results_nolabel/res_enzymes_hiddensize_128_temperature_0.8.txt",
#    "./results_nolabel/res_enzymes_hiddensize_128_temperature_0.95.txt",
#    "./results_nolabel/res_enzymes_hiddensize_128_temperature_0.9.txt",
#    "./results_nolabel/res_enzymes_hiddensize_128_temperature_1.txt",
#    "./results_nolabel/res_enzymes_hiddensize_64_temperature_0.85.txt",
#    "./results_nolabel/res_enzymes_hiddensize_64_temperature_0.8.txt",
#    "./results_nolabel/res_enzymes_hiddensize_64_temperature_0.95.txt",
#    "./results_nolabel/res_enzymes_hiddensize_64_temperature_0.9.txt",
#    "./results_nolabel/res_enzymes_hiddensize_64_temperature_1.txt",
#]

files = [
    "./results_nolabel/res_proteins_hiddensize_128_temperature_0.85.txt",
    "./results_nolabel/res_proteins_hiddensize_128_temperature_0.8.txt",
    "./results_nolabel/res_proteins_hiddensize_128_temperature_0.95.txt",
    "./results_nolabel/res_proteins_hiddensize_128_temperature_0.9.txt",
    "./results_nolabel/res_proteins_hiddensize_128_temperature_1.txt",
    "./results_nolabel/res_proteins_hiddensize_64_temperature_0.85.txt",
    "./results_nolabel/res_proteins_hiddensize_64_temperature_0.8.txt",
    "./results_nolabel/res_proteins_hiddensize_64_temperature_0.95.txt",
    "./results_nolabel/res_proteins_hiddensize_64_temperature_0.9.txt",
    "./results_nolabel/res_proteins_hiddensize_64_temperature_1.txt",
]

# Clustering MMDが最小の値を保持する変数
min_clustering_mmd = float("inf")
min_file_info = {}

# 正規表現パターン
patterns = {
    "peak_memory": r"Peak memory allocated: (\d+)",
    "total_learning_time": r"Total learning time: ([\d\.]+) seconds",
    "degree_mmd": r"Degree MMD:\s+([\d\.]+)",
    "clustering_mmd": r"Clustering MMD:\s+([\d\.]+)",
    "orbit_mmd": r"Orbit MMD\(4-node motif, 11 classes\): ([\d\.]+)",
    "graph_kernel_mmd": r"Graph KernelMMD:\s+([\d\.]+)",
}

# 各ファイルを処理
for file in files:
    try:
        with open(file, "r") as f:
            content = f.read()

        # 各値を抽出
        peak_memories = [int(match) for match in re.findall(patterns["peak_memory"], content)]
        total_learning_times = [float(match) for match in re.findall(patterns["total_learning_time"], content)]
        degree_mmd = float(re.search(patterns["degree_mmd"], content).group(1))
        clustering_mmd = float(re.search(patterns["clustering_mmd"], content).group(1))
        orbit_mmd = float(re.search(patterns["orbit_mmd"], content).group(1))
        graph_kernel_mmd = float(re.search(patterns["graph_kernel_mmd"], content).group(1))

        max_peak_memory = max(peak_memories) if peak_memories else 0
        max_total_learning_time = max(total_learning_times) if total_learning_times else 0.0

        # Clustering MMDが最小のファイルを更新
        if clustering_mmd < min_clustering_mmd:
            min_clustering_mmd = clustering_mmd
            min_file_info = {
                "file": file,
                "peak_memory": max_peak_memory,
                "total_learning_time": max_total_learning_time,
                "degree_mmd": degree_mmd,
                "clustering_mmd": clustering_mmd,
                "orbit_mmd": orbit_mmd,
                "graph_kernel_mmd": graph_kernel_mmd,
            }

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# CSVに結果を出力
if min_file_info:
    output_file = "min_clustering_mmd_results.csv"
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "file", "peak_memory", "total_learning_time",
            "degree_mmd", "clustering_mmd", "orbit_mmd", "graph_kernel_mmd"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(min_file_info)

    print(f"Results saved to {output_file}")
else:
    print("No valid files found.")