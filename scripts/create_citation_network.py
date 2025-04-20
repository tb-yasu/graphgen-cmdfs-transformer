import pandas as pd
import numpy as np
from scipy import sparse, io
import networkx as nx

def load_citeseer_raw(file_path: str):
    """Citeseerの生データを読み込む"""
    # データを一行ずつ読み込んで処理
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 最初の7つのフィールドと残りのタイトル部分に分割
            parts = line.strip().split('|', 7)
            if len(parts) == 8:  # 有効な行のみ処理
                # 各フィールドの空白を削除
                parts = [p.strip() for p in parts]
                rows.append(parts)
            else:
                print(f"Warning: Skipping invalid line: {line.strip()}")
    
    # データフレームに変換
    columns = ['author_id', 'author_cluster_id', 'normalized_author', 'full_author',
              'author_number_in_paper', 'paper_id', 'paper_cluster_id', 'title']
    df = pd.DataFrame(rows, columns=columns)
    
    # 数値型のカラムを変換
    df['author_cluster_id'] = pd.to_numeric(df['author_cluster_id'])
    df['author_number_in_paper'] = pd.to_numeric(df['author_number_in_paper'])
    
    # paper_idとpaper_cluster_idは文字列として保持
    # paper_idに含まれるハイフンを処理するため
    df['paper_id'] = df['paper_id'].astype(str)
    df['paper_cluster_id'] = df['paper_cluster_id'].astype(str)
    
    return df

def create_citation_networks(df: pd.DataFrame):
    """異なる種類のネットワークを構築する"""
    # 1. 論文の引用ネットワーク
    paper_ids = sorted(set(df['paper_id'].unique()))
    paper_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
    n_papers = len(paper_ids)
    
    paper_network = sparse.lil_matrix((n_papers, n_papers), dtype=np.int8)
    paper_labels = [None] * n_papers
    
    # paper_cluster_idでグループ化して処理
    for cluster_id in df['paper_cluster_id'].unique():
        cluster_papers = df[df['paper_cluster_id'] == cluster_id]['paper_id'].unique()
        if len(cluster_papers) > 1:
            for i, pid1 in enumerate(cluster_papers):
                idx1 = paper_to_idx[pid1]
                # タイトルを保存
                if paper_labels[idx1] is None:
                    paper_labels[idx1] = df[df['paper_id'] == pid1]['title'].iloc[0]
                for pid2 in cluster_papers[i+1:]:
                    idx2 = paper_to_idx[pid2]
                    paper_network[idx1, idx2] = 1
                    paper_network[idx2, idx1] = 1
    
    # 2. 著者の共著ネットワーク
    author_ids = sorted(set(df['author_id'].unique()))
    author_to_idx = {aid: idx for idx, aid in enumerate(author_ids)}
    n_authors = len(author_ids)
    
    author_network = sparse.lil_matrix((n_authors, n_authors), dtype=np.int8)
    author_labels = [None] * n_authors
    
    for pid in df['paper_id'].unique():
        paper_authors = df[df['paper_id'] == pid]
        authors = paper_authors['author_id'].unique()
        for i, aid1 in enumerate(authors):
            idx1 = author_to_idx[aid1]
            if author_labels[idx1] is None:
                author_labels[idx1] = paper_authors[paper_authors['author_id'] == aid1]['normalized_author'].iloc[0]
            for aid2 in authors[i+1:]:
                idx2 = author_to_idx[aid2]
                author_network[idx1, idx2] = 1
                author_network[idx2, idx1] = 1
    
    return {
        'paper_network': paper_network.tocsr(),
        'paper_labels': paper_labels,
        'paper_to_idx': paper_to_idx,
        'author_network': author_network.tocsr(),
        'author_labels': author_labels,
        'author_to_idx': author_to_idx
    }

def save_networks(networks: dict, output_path: str):
    """ネットワークデータを.matファイルとして保存"""
    # Noneを空文字列に変換
    save_dict = networks.copy()
    save_dict['paper_labels'] = ['' if label is None else label for label in networks['paper_labels']]
    save_dict['author_labels'] = ['' if label is None else label for label in networks['author_labels']]
    
    # numpy配列に変換
    save_dict['paper_labels'] = np.array(save_dict['paper_labels'], dtype=object)
    save_dict['author_labels'] = np.array(save_dict['author_labels'], dtype=object)
    
    # ID mappingを文字列のキーを持つ配列に変換
    paper_mapping = np.array([(str(k), v) for k, v in networks['paper_to_idx'].items()], 
                           dtype=[('key', 'O'), ('value', 'i4')])
    author_mapping = np.array([(str(k), v) for k, v in networks['author_to_idx'].items()],
                            dtype=[('key', 'O'), ('value', 'i4')])
    
    save_dict['paper_to_idx'] = paper_mapping
    save_dict['author_to_idx'] = author_mapping
    
    io.savemat(output_path, save_dict)

# メイン処理
if __name__ == "__main__":
    input_file = "/home/tabei/Dat/graph_dataset/citeseer-entity-resolution/citeseer-mrdm05.dat"
    output_file = "citeseer.mat"
    
    print("Loading data...")
    df = load_citeseer_raw(input_file)
    
    print("Creating networks...")
    networks = create_citation_networks(df)
    
    print("Saving networks...")
    save_networks(networks, output_file)
    
    # 基本的な統計情報を表示
    print("\nNetwork statistics:")
    print(f"Paper network: {networks['paper_network'].shape[0]} nodes, {networks['paper_network'].nnz//2} edges")
    print(f"Author network: {networks['author_network'].shape[0]} nodes, {networks['author_network'].nnz//2} edges")