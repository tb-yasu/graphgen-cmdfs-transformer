import networkx as nx
import random
import os

def random_partition(n: int, k: int) -> list[int]:
    """
    正の整数 n を k 個の正の整数にランダムに分割したリストを返す。
    例: n=10, k=3 -> [3, 2, 5] など。
    """
    # とりあえず各ブロックに1つずつ割り当てる
    sizes = [1] * k
    # 残り (n - k) をランダムに割り振る
    for _ in range(n - k):
        sizes[random.randrange(k)] += 1
    return sizes

def generate_community_graph(p_in_range: tuple[float, float],
                             p_out_range: tuple[float, float]) -> nx.Graph:
    """
    コミュニティ構造をもつランダムグラフを生成し、NetworkX グラフとして返す。
    - 全体のノード数: 12～20 の範囲からランダムに選ぶ
    - コミュニティ数: 2～5 の範囲からランダムに選ぶ
    - コミュニティ内エッジ生成確率: p_in_range から一様ランダム
    - コミュニティ間エッジ生成確率: p_out_range から一様ランダム
    """
    # 1) 全体ノード数 N を 12～20 からランダムに選ぶ
    total_nodes = random.randint(12, 20)

    # 2) コミュニティ数 k を 2～5 からランダムに選ぶ
    k = random.randint(2, 5)

    # 3) N を k 個にランダム分割してコミュニティサイズを決定
    community_sizes = random_partition(total_nodes, k)

    # 4) コミュニティ内 / コミュニティ間 のエッジ生成確率行列 p を作成
    p_in = random.uniform(*p_in_range)    # コミュニティ内エッジ生成確率
    p_out = random.uniform(*p_out_range)  # コミュニティ間エッジ生成確率

    p = [[0.0]*k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                p[i][j] = p_in   # 同じコミュニティ
            else:
                p[i][j] = p_out  # 別コミュニティ

    # 5) SBM を使ってランダムグラフ生成
    G = nx.stochastic_block_model(community_sizes, p, seed=random.randint(0,999999))
    return G

def save_gspan_format(G: nx.Graph, graph_id: int, f):
    """
    与えられた NetworkX グラフ G を gSpan 形式でファイルに追記する。
    graph_id は gSpan 上でのグラフ識別子。
    f は書き込み先ファイルオブジェクト。
    """
    # Node ごとに 0,1,2,... の連番を振る
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    
    # グラフの開始
    f.write(f"t # {graph_id}\n")

    # ノード出力 (ラベルは 0 として簡単化)
    for node in G.nodes():
        f.write(f"v {node_map[node]} 0\n")

    # エッジ出力 (無向グラフなので、NetworkXのedges()で1回のみ取得)
    for u, v in G.edges():
        f.write(f"e {node_map[u]} {node_map[v]} 0\n")

def main():
    """
    100 個の SBM グラフ (ノード数 12～20) を生成し、
    gSpan 形式ファイル (community_small_gspan.txt) に連続して出力する。
    """
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "community_small_gspan.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        # グラフを 100 個作成して書き出す
        for i in range(1, 101):
            # コミュニティ内エッジ生成確率 [0.5, 0.9], コミュニティ間 [0.01, 0.05]
            G = generate_community_graph(
                p_in_range=(0.5, 0.9),
                p_out_range=(0.01, 0.05)
            )
            # gSpan形式で書き出し
            save_gspan_format(G, i, f)

    print(f"Saved gSpan dataset to: {output_file}")

if __name__ == "__main__":
    main()