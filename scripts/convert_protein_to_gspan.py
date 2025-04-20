import os

def read_proteins_full_dataset(
    base_dir="PROTEINS_full",
    A_file="PROTEINS_full_A.txt",
    graph_indicator_file="PROTEINS_full_graph_indicator.txt",
    graph_labels_file="PROTEINS_full_graph_labels.txt",
    node_labels_file="PROTEINS_full_node_labels.txt",
    edge_labels_file=None,    # 必要に応じて追加
    output_file="output_gspan.txt"
):
    """
    PROTEINS_full データセットを読み込み、gSpan 形式で出力する。
    
    Parameters
    ----------
    base_dir : str
        データが置いてあるディレクトリのパス
    A_file : str
        エッジ情報 (row, col) が書かれたファイル名
    graph_indicator_file : str
        各ノードがどのグラフに属するかの情報
    graph_labels_file : str
        グラフのラベル一覧 (今回は gSpan 出力には必須ではない)
    node_labels_file : str
        ノードのラベル一覧 (gSpan のノードラベルとして使用)
    edge_labels_file : str or None
        辺のラベル一覧 (必要なら使用)
    output_file : str
        gSpan 形式で書き出す出力ファイル
    """

    # ファイルのパスを作成
    path_A = os.path.join(base_dir, A_file)
    path_indicator = os.path.join(base_dir, graph_indicator_file)
    path_graph_labels = os.path.join(base_dir, graph_labels_file)
    path_node_labels = os.path.join(base_dir, node_labels_file) if node_labels_file else None
    path_edge_labels = os.path.join(base_dir, edge_labels_file) if edge_labels_file else None

    # ==== 1. graph_indicator の読み込み (ノードID -> グラフID) ====
    #     ノードID i は 1-based, グラフID も 1-based
    node_to_graph = {}
    with open(path_indicator, "r") as f:
        for i, line in enumerate(f, start=1):
            graph_id = int(line.strip())
            node_to_graph[i] = graph_id
    
    # グラフ数 N を求める（graph_labels_file があるなら行数でも確認できる）
    # ここでは最大の graph_id を N とする
    N = max(node_to_graph.values())

    # ==== 2. node_labels の読み込み (ノードID -> ラベル) ====
    #     ノードID i は 1-based
    node_labels = {}
    if path_node_labels is not None and os.path.exists(path_node_labels):
        with open(path_node_labels, "r") as f:
            for i, line in enumerate(f, start=1):
                label = line.strip()
                node_labels[i] = label
    else:
        # ノードラベルが無い場合は、すべて同じラベルにするなど適宜
        # ここでは "0" とする
        for i in range(1, len(node_to_graph) + 1):
            node_labels[i] = "0"

    # ==== 3. エッジ情報 (PROTEINS_full_A.txt) の読み込み ====
    #     グラフごとにエッジを仕分けるために辞書構造を使う
    #     edges_by_graph[g] = set( (min(u,v), max(u,v)) のタプル ) とする
    edges_by_graph = {g: set() for g in range(1, N+1)}

    # edge_labels を使う場合はここで読み込む（必要なら）
    edge_labels = None
    if path_edge_labels and os.path.exists(path_edge_labels):
        edge_labels = []
        with open(path_edge_labels, "r") as f:
            for line in f:
                edge_labels.append(line.strip())
        # edge_labels の i 番目が A_file の i 行目に対応する

    with open(path_A, "r") as f:
        for edge_index, line in enumerate(f):
            row, col = line.strip().split(",")
            row, col = int(row), int(col)

            # row の属するグラフ = col の属するグラフ になるはず
            g = node_to_graph[row]
            # 念のためチェック
            if node_to_graph[col] != g:
                raise ValueError("不正: 一つのエッジが異なるグラフIDにまたがっている")

            # 無向グラフを想定し (min, max) で持つ
            u, v = min(row, col), max(row, col)
            # エッジラベル（もしあれば）
            if edge_labels is not None:
                # edge_labels[edge_index] が対応ラベル
                edges_by_graph[g].add((u, v, edge_labels[edge_index]))
            else:
                # 辺のラベルを 1 (あるいは 0) とするなど適宜
                edges_by_graph[g].add((u, v, "1"))

    # ==== 4. gSpan フォーマットで出力 ====
    # グラフIDは gSpan では 0-based にすることが多いので調整する
    # ただし "t # <元のID>" でも良い場合はそのままでもよい
    with open(output_file, "w") as fout:
        for g in range(1, N+1):
            # グラフ開始: t # g-1 (0-based の場合)
            fout.write(f"t # {g-1}\n")

            # --- グラフ g のノードたちを抽出 ---
            # node_to_graph から g に属するノード一覧を取り出す
            nodes_in_g = [nid for nid in node_to_graph if node_to_graph[nid] == g]
            nodes_in_g.sort()  # 小さい順に並べる (1-based)

            # ノードをローカルID (0-based) にマッピングする
            #   global_node_id -> local_node_id
            local_id_map = {}
            for local_id, nid in enumerate(nodes_in_g):
                local_id_map[nid] = local_id
                label = node_labels[nid]
                label = int(label) + 1
                fout.write(f"v {local_id} {label}\n")

            # --- グラフ g のエッジを書き出し ---
            # edges_by_graph[g] に (u, v, lbl) が入っている
            for (u, v, lbl) in edges_by_graph[g]:
                # ローカルIDに変換
                lu = local_id_map[u]
                lv = local_id_map[v]
                fout.write(f"e {lu} {lv} {lbl}\n")


def main():
    # 実際に呼び出す例
    base_dir = "PROTEINS_full"  # データが置いてあるディレクトリ
    read_proteins_full_dataset(
        base_dir="/home/tabei/prog/python/graph-generation/dataset/PROTEINS_full/",
        A_file="PROTEINS_full_A.txt",
        graph_indicator_file="PROTEINS_full_graph_indicator.txt",
        graph_labels_file="PROTEINS_full_graph_labels.txt",
        node_labels_file="PROTEINS_full_node_labels.txt",
        edge_labels_file=None,  # 必要に応じてファイル名をセット
        output_file="output_gspan.txt"
    )
    print("gSpan 形式ファイルを出力しました。")

if __name__ == "__main__":
    main()
