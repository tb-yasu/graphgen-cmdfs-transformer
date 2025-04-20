import networkx as nx

def parse_gspan_multiple(file_path):
    """
    複数のgSpanフォーマットのグラフを解析し、NetworkXのグラフのリストを返す。
    """
    graphs = []
    G = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 't':
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()
            elif parts[0] == 'v' and G is not None:
                vertex_id = int(parts[1])
                label = parts[2]
                G.add_node(vertex_id, label=label)
            elif parts[0] == 'e' and G is not None:
                src = int(parts[1])
                tgt = int(parts[2])
                label = parts[3]
                G.add_edge(src, tgt, label=label)
        if G is not None:
            graphs.append(G)
    return graphs

def check_connected_multiple(file_path):
    """
    gSpanフォーマットのファイル内のすべてのグラフが連結かどうかをチェックし、結果を返す。
    """
    graphs = parse_gspan_multiple(file_path)
    results = {}
    for idx, G in enumerate(graphs):
        results[idx] = nx.is_connected(G)
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='gSpanフォーマットのグラフが連結かどうかをチェックするプログラム（複数グラフ対応）')
    parser.add_argument('file', help='gSpanフォーマットのファイルパス')
    args = parser.parse_args()

    counter = 0
    total = 0 
    try:
        results = check_connected_multiple(args.file)
        for graph_id, connected in results.items():
            total += 1
            if connected:
                pass
            else:
                print(f"グラフ {graph_id} は連結ではありません。")
                counter += 1
#            status = "連結です。" if connected else "連結ではありません。"
#            print(f"グラフ {graph_id}: {status}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

    print(f"連結ではないグラフの数: {counter}")
    print(f"グラフの総数: {total}")
    print(f"連結ではないグラフの割合: {counter / total:.2f}")
