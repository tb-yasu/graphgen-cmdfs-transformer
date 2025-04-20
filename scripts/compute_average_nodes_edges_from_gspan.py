#!/usr/bin/env python3
#!/usr/bin/env python3
from collections import defaultdict
import math

def calculate_average_nodes_edges_and_acc(gspan_file_path):
    """
    gSpanフォーマットのファイルを読み込み、
      - グラフごとのノード数とエッジ数
      - グラフごとの平均クラスタリング係数(ACC: Average Clustering Coefficient)
    を求める関数

    返り値:
        (全グラフの平均ノード数, 全グラフの平均エッジ数, 全グラフの平均クラスタリング係数)
    """
    with open(gspan_file_path, 'r') as f:
        graphs_node_counts = []
        graphs_edge_counts = []
        graphs_accs = []  # グラフごとのACCを格納

        # グラフ構築用
        current_nodes = set()
        current_edges = set()
        # 隣接リスト(無向グラフ想定): node -> set(neighbors)
        adjacency = defaultdict(set)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 新しいグラフの開始行: t # <id>
            if line.startswith('t #'):
                # 以前のグラフを集計して格納
                if len(current_nodes) > 0 or len(current_edges) > 0:
                    # ノード数・エッジ数
                    graphs_node_counts.append(len(current_nodes))
                    graphs_edge_counts.append(len(current_edges))

                    # このグラフの平均クラスタリング係数(ACC)を計算
                    acc = compute_average_clustering_coefficient(adjacency)
                    graphs_accs.append(acc)

                # 新しいグラフに向けて初期化
                current_nodes = set()
                current_edges = set()
                adjacency = defaultdict(set)

            # ノード行: v <id> <label>
            elif line.startswith('v '):
                parts = line.split()
                vertex_id = int(parts[1])
                current_nodes.add(vertex_id)

            # エッジ行: e <id1> <id2> <label>
            elif line.startswith('e '):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                # 無向グラフとして扱うので、順序を正規化
                edge = tuple(sorted([v1, v2]))
                if edge not in current_edges:
                    current_edges.add(edge)
                    # 隣接リストにも登録
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)

        # ファイル終端に来たとき、最後のグラフも集計
        if len(current_nodes) > 0 or len(current_edges) > 0:
            graphs_node_counts.append(len(current_nodes))
            graphs_edge_counts.append(len(current_edges))
            # このグラフのACC
            acc = compute_average_clustering_coefficient(adjacency)
            graphs_accs.append(acc)

    num_graphs = len(graphs_node_counts)
    if num_graphs == 0:
        return 0.0, 0.0, 0.0

    avg_nodes = sum(graphs_node_counts) / num_graphs
    avg_edges = sum(graphs_edge_counts) / num_graphs
    avg_acc = sum(graphs_accs) / num_graphs

    return avg_nodes, avg_edges, avg_acc


def compute_average_clustering_coefficient(adjacency):
    """
    与えられた隣接リスト(無向グラフ)から、グラフの平均クラスタリング係数(ACC)を計算する。
    
    adjacency: dict(node -> set of neighbors)

    1つのノードvに対する局所クラスタリング係数C_vは
      C_v = (# of edges among neighbors of v) / { deg(v)*(deg(v)-1)/2 }
    ただしdeg(v)=1以下の場合はC_v=0とする。
    """
    if not adjacency:
        return 0.0

    total_clustering = 0.0
    nodes = list(adjacency.keys())

    for v in nodes:
        neighbors = adjacency[v]
        deg = len(neighbors)
        # 次数が 0 や 1 だと三角形は作れないので、C_v=0
        if deg < 2:
            continue
        
        # 隣接ノード同士の辺を数える
        # ここでは単純に二重ループで数える（O(deg^2））
        edges_among_neighbors = 0
        neighbor_list = list(neighbors)
        for i in range(deg):
            for j in range(i+1, deg):
                u1 = neighbor_list[i]
                u2 = neighbor_list[j]
                # u2 が u1 の隣接セットに含まれるかチェック
                if u2 in adjacency[u1]:
                    edges_among_neighbors += 1

        # 最大可能辺数: deg(v)*(deg(v)-1)/2
        possible_edges = deg * (deg - 1) / 2.0
        c_v = edges_among_neighbors / possible_edges
        total_clustering += c_v

    # 全ノードに対する平均
    # 「隣接リストに登場するノード数」として計算
    #   → 次数0/1のノードも 0 として含まれているのでOK
    #     (上のループでは continue してるが、値は0のまま足さないため)
    acc = total_clustering / len(nodes)
    return acc


def main():
    # gSpanフォーマットのファイルを指定
    input_file = "/home/tabei/Dat/software/GxRNN/datasets/LINCS/mcf7.gsp"

    avg_nodes, avg_edges, avg_acc = calculate_average_nodes_edges_and_acc(input_file)

    # 結果表示
    print(f"Average number of nodes: {avg_nodes:.2f}")
    print(f"Average number of edges: {avg_edges:.2f}")
    print(f"Average clustering coefficient: {avg_acc:.4f}")  # 小数点4桁表示など

if __name__ == "__main__":
    main()
