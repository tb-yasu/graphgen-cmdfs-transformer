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

# MMDの最小値を保持する変数
min_values = {
    "degree_mmd": float("inf"),
    "clustering_mmd": float("inf"),
    "orbit_mmd": float("inf"),
    "graph_kernel_mmd": float("inf"),
    "files": {
        "degree_mmd": None,
        "clustering_mmd": None,
        "orbit_mmd": None,
        "graph_kernel_mmd": None,
    }
}

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
        degree_mmd = float(re.search(patterns["degree_mmd"], content).group(1))
        clustering_mmd = float(re.search(patterns["clustering_mmd"], content).group(1))
        orbit_mmd = float(re.search(patterns["orbit_mmd"], content).group(1))
        graph_kernel_mmd = float(re.search(patterns["graph_kernel_mmd"], content).group(1))

        # 各MMDの最小値を更新
        if degree_mmd < min_values["degree_mmd"]:
            min_values["degree_mmd"] = degree_mmd
            min_values["files"]["degree_mmd"] = file

        if clustering_mmd < min_values["clustering_mmd"]:
            min_values["clustering_mmd"] = clustering_mmd
            min_values["files"]["clustering_mmd"] = file

        if orbit_mmd < min_values["orbit_mmd"]:
            min_values["orbit_mmd"] = orbit_mmd
            min_values["files"]["orbit_mmd"] = file

        if graph_kernel_mmd < min_values["graph_kernel_mmd"]:
            min_values["graph_kernel_mmd"] = graph_kernel_mmd
            min_values["files"]["graph_kernel_mmd"] = file

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 結果をプロット
mmd_types = ["degree_mmd", "clustering_mmd", "orbit_mmd", "graph_kernel_mmd"]
mmd_values = [min_values[mmd_type] for mmd_type in mmd_types]

# CSVに結果を出力
output_file = "min_mmd_results.csv"
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # ヘッダー行を書き込む
    writer.writerow(["degree_mmd", "clustering_mmd", "orbit_mmd", "graph_kernel_mmd"])
    # 最小値の行を書き込む
    writer.writerow(mmd_values)

print(f"Results saved to {output_file}")
