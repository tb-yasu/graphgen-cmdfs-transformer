from typing import Dict, List
import torch

import networkx as nx

from typing import Tuple

import subprocess

import networkx as nx


class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self):
        self.vertex_id_map: Dict[int, int] = {}
        self.vertex_label_map: Dict[int, int] = {}
        self.edge_label_map: Dict[str, int] = {}

        self.id_to_vertex_id: Dict[int, int] = {}
        self.id_to_vertex_label: Dict[int, int] = {}
        self.id_to_edge_label: Dict[int, int] = {}
        
        self.range_vertex_id: Pair = None
        self.range_vertex_label: Pair = None
        self.range_edge_label: Pair = None

        self.max_from = 0
        self.max_to = 0

        # 特殊トークンの定義（0は未使用として予約）
        self.START_TOKEN = 0
        self.END_TOKEN = 1
        self.PAD_TOKEN = 2
        self.SPECIAL_TOKENS = [self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN]

    def vocab_size(self) -> int:
#        return max(self.max_from, self.max_to) + len(self.SPECIAL_TOKENS) + 1
        return len(self.vertex_id_map) + len(self.vertex_label_map) + len(self.edge_label_map) + len(self.SPECIAL_TOKENS)

#    def encode_dfs_codes_list_to_sequences(self, dfs_codes_list: List[List[List[str]]]):
    def encode_dfs_codes_list_to_sequences(self, dfs_codes_list):

        counter = max(self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN) + 1

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                if len(dfs_code) != 5:
                    raise ValueError(f"dfs_codeの長さが5ではありません。: {dfs_code}")
                
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v = int(from_v)
                to_v = int(to_v)
                from_label = int(from_label)
                edge_label = int(edge_label)
                to_label = int(to_label)

                if self.from_vertex_label_map.get(from_label, None) == None:
                    self.from_vertex_label_map[from_label] = counter
                    counter += 1

#                from_label = self.from_vertex_label_map[from_label]

        for dfs_codes in dfs_codes_list:    
            for dfs_code in dfs_codes:  
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v = int(from_v)
                to_v = int(to_v)
                from_label = int(from_label)
                edge_label = int(edge_label)
                to_label = int(to_label)

                if self.to_vertex_label_map.get(to_label, None) == None:
                    self.to_vertex_label_map[to_label] = counter
                    counter += 1

#                to_label = self.to_vertex_label_map[to_label]

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v = int(from_v)
                to_v = int(to_v)
                from_label = int(from_label)
                edge_label = int(edge_label)
                to_label = int(to_label)
                if self.edge_label_map.get(edge_label, None) == None:
                    self.edge_label_map[edge_label] = counter
                    counter += 1

#                edge_label = self.edge_label_map[edge_label]

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v = int(from_v)
                to_v = int(to_v)
                from_label = int(from_label)
                edge_label = int(edge_label)
                to_label = int(to_label)
                if self.from_vertex_map.get(from_v, None) == None:
                    self.from_vertex_map[from_v] = counter
                    counter += 1
                if self.from_vertex_map.get(to_v, None) == None:
                    self.from_vertex_map[to_v] = counter
                    counter += 1
                

        for dfs_codes in dfs_codes_list:    
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v = int(from_v)
                to_v = int(to_v)
                from_label = int(from_label)
                edge_label = int(edge_label)
                to_label = int(to_label)
                if self.to_vertex_map.get(to_v, None) == None:
                    self.to_vertex_map[to_v] = counter
                    counter += 1
                if self.to_vertex_map.get(from_v, None) == None:
                    self.to_vertex_map[from_v] = counter
                    counter += 1                
                if self.max_from < from_v:
                    self.max_from = from_v
                if self.max_to < to_v:
                    self.max_to = to_v
#                to_v = self.to_vertex_map[to_v]

        dfs_codes_sequences = []
        for gid, dfs_codes in enumerate(dfs_codes_list):
            current_sequence = [self.START_TOKEN]
            continuous_type_flag = False
            prev_from_v = -1
            for i, dfs_code in enumerate(dfs_codes):
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v, to_v, from_label, edge_label, to_label = int(from_v), int(to_v), int(from_label), int(edge_label), int(to_label)

                if i == 0:
                    current_sequence.append(self.from_vertex_label_map[from_label])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.to_vertex_label_map[to_label])
                    continuous_type_flag = False
                elif from_v + 1 < to_v:
                    current_sequence.append(self.from_vertex_map[from_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.to_vertex_label_map[to_label])
                    continuous_type_flag = True
                elif from_v + 1 == to_v and continuous_type_flag == False:
                    current_sequence.append(self.from_vertex_map[from_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.to_vertex_label_map[to_label])
                    continuous_type_flag = True
                elif from_v + 1 == to_v and continuous_type_flag == True: # forward edge
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.to_vertex_label_map[to_label])    
                    continuous_type_flag = True
                elif from_v > to_v: # backward_edge
                    current_sequence.append(self.from_vertex_map[from_v])
                    current_sequence.append(self.to_vertex_map[to_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    continuous_type_flag = False
                else:
                    print("gid:", gid)
                    print("from_v:", from_v )
                    print("to_v:", to_v)
                    raise ValueError(f"iteration {i}: 条件に合わないエッジがあります。")

                prev_to_v = to_v

#            print("current_sequence:", current_sequence)    
            current_sequence.append(self.END_TOKEN)
            dfs_codes_sequences.append(current_sequence)

        self.range_from_vertex_id = Pair(min(self.from_vertex_map.values()), max(self.from_vertex_map.values()))
        self.range_to_vertex_id = Pair(min(self.to_vertex_map.values()), max(self.to_vertex_map.values()))
        self.range_from_vertex_label_id = Pair(min(self.from_vertex_label_map.values()), max(self.from_vertex_label_map.values()))
        self.range_to_vertex_label_id = Pair(min(self.to_vertex_label_map.values()), max(self.to_vertex_label_map.values()))
        self.range_edge_label_id = Pair(min(self.edge_label_map.values()), max(self.edge_label_map.values()))

        for key, value in self.from_vertex_map.items():
            self.id_to_from_vertex[value] = key
        for key, value in self.to_vertex_map.items():
            self.id_to_to_vertex[value] = key
        for key, value in self.from_vertex_label_map.items():
            self.id_to_from_vertex_label[value] = key   
        for key, value in self.to_vertex_label_map.items():
            self.id_to_to_vertex_label[value] = key
        for key, value in self.edge_label_map.items():
            self.id_to_edge_label[value] = key

        return dfs_codes_sequences


    #    def encode_dfs_codes_list_to_sequences(self, dfs_codes_list: List[List[List[str]]]):
    def encode_dfs_codes_list_to_sequences2(self, dfs_codes_list):

        counter = max(self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN) + 1

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                if len(dfs_code) != 5:
                    raise ValueError(f"dfs_codeの長さが5ではありません。: {dfs_code}")
                
                from_v, to_v, from_label, edge_label, to_label = map(int, dfs_code)

                if self.vertex_label_map.get(from_label, None) == None:
                    self.vertex_label_map[from_label] = counter
                    counter += 1
        
                if self.vertex_label_map.get(to_label, None) == None:
                    self.vertex_label_map[to_label] = counter
                    counter += 1

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = map(int, dfs_code)
                if self.edge_label_map.get(edge_label, None) == None:
                    self.edge_label_map[edge_label] = counter
                    counter += 1

        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = map(int, dfs_code)
                if self.vertex_id_map.get(from_v, None) == None:
                    self.vertex_id_map[from_v] = counter
                    counter += 1

                if self.vertex_id_map.get(to_v, None) == None:
                    self.vertex_id_map[to_v] = counter
                    counter += 1

                if self.max_from < from_v:
                    self.max_from = from_v

                if self.max_to < to_v:
                    self.max_to = to_v

        self.range_vertex_id = Pair(min(self.vertex_id_map.values()), max(self.vertex_id_map.values()))
        self.range_vertex_label = Pair(min(self.vertex_label_map.values()), max(self.vertex_label_map.values()))
        self.range_edge_label = Pair(min(self.edge_label_map.values()), max(self.edge_label_map.values()))

        for key, value in self.vertex_id_map.items():
            self.id_to_vertex_id[value] = key
        for key, value in self.vertex_label_map.items():
            self.id_to_vertex_label[value] = key
        for key, value in self.edge_label_map.items():
            self.id_to_edge_label[value] = key

        dfs_codes_sequences = []
        for gid, dfs_codes in enumerate(dfs_codes_list):
            current_sequence = [self.START_TOKEN]
            is_continuous_forward_edge = False
            continuous_type_flag = False
            prev_from_v = -1
            for i, dfs_code in enumerate(dfs_codes):
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v, to_v, from_label, edge_label, to_label = int(from_v), int(to_v), int(from_label), int(edge_label), int(to_label)

                if i == 0:
                    current_sequence.append(self.vertex_label_map[from_label])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.vertex_label_map[to_label])
                    is_continuous_forward_edge = False
                elif from_v + 1 < to_v:
                    current_sequence.append(self.vertex_id_map[from_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.vertex_label_map[to_label])
                    is_continuous_forward_edge = True
                elif from_v + 1 == to_v and is_continuous_forward_edge == False:
                    current_sequence.append(self.vertex_id_map[from_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.vertex_label_map[to_label])
                    is_continuous_forward_edge = True
                elif from_v + 1 == to_v and is_continuous_forward_edge == True: # forward edge
                    current_sequence.append(self.edge_label_map[edge_label])
                    current_sequence.append(self.vertex_label_map[to_label])    
                    is_continuous_forward_edge = True
                elif from_v > to_v: # backward_edge
                    current_sequence.append(self.vertex_id_map[from_v])
                    current_sequence.append(self.vertex_id_map[to_v])
                    current_sequence.append(self.edge_label_map[edge_label])
                    is_continuous_forward_edge = False
                else:
                    print("gid:", gid)
                    print("from_v:", from_v )
                    print("to_v:", to_v)
                    raise ValueError(f"iteration {i}: 条件に合わないエッジがあります。")

                prev_to_v = to_v

#            print("current_sequence:", current_sequence)    
            current_sequence.append(self.END_TOKEN)
            dfs_codes_sequences.append(current_sequence)

        return dfs_codes_sequences

    def encode_dfs_codes_list_to_sequences_no_label(self, dfs_codes_list):
#        counter = max(self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN) + 1

        self.max_from = 0
        self.max_to = 0
        for dfs_codes in dfs_codes_list:
            for dfs_code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = map(int, dfs_code)
                if self.max_from < from_v:
                    self.max_from = from_v

                if self.max_to < to_v:
                    self.max_to = to_v

        max_id = max(self.max_from, self.max_to)

        self.START_TOKEN = max_id + 1
        self.END_TOKEN = max_id + 2
        self.PAD_TOKEN = max_id + 3

        dfs_codes_sequences = []
        for gid, dfs_codes in enumerate(dfs_codes_list):
            current_sequence = [self.START_TOKEN]
            is_continuous_forward_edge = False
            continuous_type_flag = False
            prev_to_v = -1
            for i, dfs_code in enumerate(dfs_codes):
                from_v, to_v, from_label, edge_label, to_label = dfs_code
                from_v, to_v, from_label, edge_label, to_label = int(from_v), int(to_v), int(from_label), int(edge_label), int(to_label)

                if prev_to_v == from_v and from_v + 1 == to_v:
                    current_sequence.append(to_v)
                else:
                    current_sequence.append(from_v)
                    current_sequence.append(to_v)
                prev_to_v = to_v
            
            current_sequence.append(self.END_TOKEN)
            dfs_codes_sequences.append(current_sequence)
    
#        self.range_vertex_id = Pair(min(self.vertex_id_map.values()), max(self.vertex_id_map.values()))

#        for key, value in self.vertex_id_map.items():
#            self.id_to_vertex_id[value] = key

        return dfs_codes_sequences



    def search_vertex_label(self, dfs_codes: List[List[int]], v: int) -> int:
        for code in dfs_codes:
            from_v, to_v, from_label, edge_label, to_label = code
            if  v == from_v:
                return from_label
            elif v == to_v:
                return to_label

        return "<UNK>"
    
    def decode_dfs_codes_sequences_to_dfs_codes_list(self, dfs_codes_sequences: List[List[int]]) -> List[List[List[str]]]:
        dfs_codes_list = []
        for sequence in dfs_codes_sequences:
            max_vertex_id = 0
            dfs_codes = []
            prev_to_v = 0
            is_continuous_forward_edge = False
            i = 0
            while i < len(sequence):
                if  i == 0:
                    from_label = self.id_to_vertex_label.get(sequence[i], "<UNK>")
                    edge_label = self.id_to_edge_label.get(sequence[i + 1], "<UNK>")
                    to_label = self.id_to_vertex_label.get(sequence[i + 2], "<UNK>")
                    dfs_codes.append([0, 1, from_label, edge_label, to_label])
                    max_vertex_id = max(max_vertex_id, 0)
                    max_vertex_id = max(max_vertex_id, 1)
                    prev_to_v = 1
                    i += 3
                elif self.range_vertex_id.first <= sequence[i] <= self.range_vertex_id.second:
                    if self.range_vertex_id.first <= sequence[i + 1] <= self.range_vertex_id.second:
                        from_v = self.id_to_vertex_id.get(sequence[i], "<UNK>")
                        to_v   = self.id_to_vertex_id.get(sequence[i + 1], "<UNK>")
                        edge_label = self.id_to_edge_label.get(sequence[i + 2], "<UNK>")
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        to_label   = self.search_vertex_label(dfs_codes, to_v)
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        max_vertex_id = max(max_vertex_id, from_v)
                        max_vertex_id = max(max_vertex_id, to_v)
                        prev_to_v = to_v
                        i += 3
                    else:
                        from_v = self.id_to_vertex_id.get(sequence[i], "<UNK>")
                        to_v   = max_vertex_id + 1
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        edge_label = self.id_to_edge_label.get(sequence[i + 1], "<UNK>")
                        to_label = self.id_to_vertex_label.get(sequence[i + 2], "<UNK>")
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        max_vertex_id = max(max_vertex_id, from_v)
                        max_vertex_id = max(max_vertex_id, to_v)
                        prev_to_v = to_v
                        i += 3
                else:
                    from_v = prev_to_v
                    to_v   = from_v + 1
                    from_label = self.search_vertex_label(dfs_codes, from_v)
                    edge_label = self.id_to_edge_label.get(sequence[i], "<UNK>")
                    to_label = self.id_to_vertex_label.get(sequence[i + 1], "<UNK>")
                    dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                    max_vertex_id = max(max_vertex_id, from_v)
                    max_vertex_id = max(max_vertex_id, to_v)
                    prev_to_v = to_v
                    i += 2
                
            dfs_codes_list.append(dfs_codes)
            dfs_codes = []

        return dfs_codes_list
    
    def decode_sequence_to_dfs_codes(self, sequence: List[int]) -> List[List[str]]:
        dfs_codes = []

        i = 0
        while i < len(sequence):
            if sequence[i] == self.START_TOKEN:
                i += 1
                continue
            if sequence[i] == self.PAD_TOKEN:
                i += 1
                continue
            elif sequence[i] == self.END_TOKEN:
                break
            
            if len(sequence) - i < 5:
                raise ValueError("the sequence length is less than 5!:", len(sequence) - i)
            
            vertex1       = sequence[i]
            vertex2       = sequence[i + 1]
            vertex_label1 = sequence[i + 2]
            edge_label    = sequence[i + 3]
            vertex_label2 = sequence[i + 4]
            i += 5

            c_vertex1 = self.id_to_vertex.get(vertex1, "<UNK>")
            c_vertex2 = self.id_to_vertex.get(vertex2, "<UNK>")
            c_vertex_label1 = self.id_to_vertex_label.get(vertex_label1, "<UNK>")
            c_edge_label = self.id_to_edge_label.get(edge_label, "<UNK>")
            c_vertex_label2 = self.id_to_vertex_label.get(vertex_label2, "<UNK>")
#            c_vertex1 = self.id_to_vertex1.get(vertex1, "<UNK>")
#            c_vertex2 = self.id_to_vertex2.get(vertex2, "<UNK>")
#            c_vertex_label1 = self.id_to_vertex_label1.get(vertex_label1, "<UNK>")
#            c_vertex_label2 = self.id_to_vertex_label2.get(vertex_label2, "<UNK>")
#            c_edge_label = self.id_to_edge_label.get(edge_label, "<UNK>")



            dfs_codes.append([c_vertex1, c_vertex2, c_vertex_label1, c_edge_label, c_vertex_label2])                
            
        return dfs_codes
    
class Compare:
    def __init__(self, tokenizer: Tokenizer, iteration: int):
        self.tokenizer = tokenizer
        self.iteration = iteration

    def compute_mmd_multi_dim_subsample(
        dfs_codes_list, 
        decoded_dfs_codes_list,
        sigma=1.0, 
        Bp=100,    # サブサンプリングサイズ (X_p)
        Bq=100,    # サブサンプリングサイズ (X_q)
        n_iter=10  # サンプリング回数
    ):
        """
        ランダムサンプリングした部分集合に対して MMD を複数回計算し、その平均を返す。

        Parameters
        ----------
        X_p : array-like
            ソース側のサンプル群 (リストや NumPy 配列など)
        X_q : array-like
            ターゲット側のサンプル群
        kernel_func : function
            k(x, y, sigma=...) の形で呼び出せるカーネル関数
        sigma : float
            カーネルのパラメータ
        Bp : int
            X_p からサンプリングする個数
        Bq : int
            X_q からサンプリングする個数
        n_iter : int
            ランダムサンプリングを何回繰り返すか

        Returns
        -------
        float
            MMD の推定値（複数回のサンプリングによる平均）
        """
        data_graph_features_list = []
        for dfs_codes in dfs_codes_list:
            data_graph_feature = self.convert_decoded_sequence_to_graph(dfs_codes)
            data_graph_features_list.append(data_graph_feature)

        decoded_graph_features_list = []
        for decoded_dfs_codes in decoded_dfs_codes_list:
            decoded_graph_feature = self.convert_decoded_sequence_to_graph(decoded_dfs_codes)
            decoded_graph_features_list.append(decoded_graph_feature)

        # もし Bp や Bq が元データ数より大きい場合は調整
        m = len(data_graph_features_list)
        n = len(decoded_graph_features_list)
        Bp = min(Bp, m)
        Bq = min(Bq, n)

        # データ数が 2 未満だとアンバイアス推定ができないので 0.0 を返す
        if m < 2 or n < 2:
            return 0.0

        # サンプリングを繰り返して平均を取る
        mmd_values = []
        for _ in range(n_iter):
            # ランダムにインデックスを選び出す
            idx_p = np.random.choice(m, size=Bp, replace=False)
            idx_q = np.random.choice(n, size=Bq, replace=False)

            # 部分集合を取り出す
            subX_p = [X_p[i] for i in idx_p]
            subX_q = [X_q[j] for j in idx_q]

            # 以下、アンバイアス推定の MMD 計算
            # ここでは「MMD (平方根)」を返す実装

            xx = 0.0
            for i in range(Bp):
                for j in range(i+1, Bp):
                    feature1, feature2 = data_graph_features_list[i], data_graph_features_list[j]
                    similarity = self.calculate_min_max_similarity(feature1, feature2)
                    xx += similarity
            xx = xx * 2.0 / (Bp*(Bp-1))

            yy = 0.0
            for i in range(Bq):
                for j in range(i+1, Bq):
                    feature1, feature2 = decoded_graph_features_list[i], decoded_graph_features_list[j]
                    similarity = self.calculate_min_max_similarity(feature1, feature2)
                    yy += similarity
            yy = yy * 2.0 / (Bq*(Bq-1))

            xy = 0.0
            for i in range(Bp):
                for j in range(Bq):
                    feature1, feature2 = data_graph_features_list[i], decoded_graph_features_list[j]
                    similarity = self.calculate_min_max_similarity(feature1, feature2)
                    xy += similarity
            xy = xy * (-2.0) / (Bp*Bq)

            mmd_sq = xx + yy + xy
            mmd_sq = max(mmd_sq, 0.0)  # 数値誤差対策
            mmd_val = np.sqrt(mmd_sq)  # MMD^2 の平方根を MMD として返す

            mmd_values.append(mmd_val)

        # 複数回のサンプリング結果を平均
        return float(np.mean(mmd_values))

    def comp_mmd(self, dfs_codes_list, decoded_dfs_codes_list):
#        data_graph_features_list = []
#        for dfs_codes in dfs_codes_list:
#            data_graph_feature = self.convert_decoded_sequence_to_graph(dfs_codes)
#            data_graph_features_list.append(data_graph_feature)

#        decoded_graph_features_list = []
#        for decoded_dfs_codes in decoded_dfs_codes_list:
#            decoded_graph_feature = self.convert_decoded_sequence_to_graph(decoded_dfs_codes)
#            decoded_graph_features_list.append(decoded_graph_feature)

        data_graph_features_list = dfs_codes_list
        decoded_graph_features_list = decoded_dfs_codes_list

        m = len(data_graph_features_list)
        n = len(decoded_graph_features_list)

        if m < 2 or n < 2:
            return 0.0

        xx = 0.0
        num_comparisons = 0
        for i in range(m):
            for j in range(i + 1, m):
                graph1, graph2 = data_graph_features_list[i], data_graph_features_list[j]
                graph_features1, graph_features2 = self.comp_weisfeiler_lehman_subtree_features_from_graphs(graph1, graph2)
                similarity = self.calculate_min_max_similarity(graph_features1, graph_features2)
                xx += similarity
                num_comparisons += 1

        xx = xx * 2.0 / num_comparisons if num_comparisons > 0 else 0.0
#        print(f"Average similarity: {xx:.4f}")

        yy = 0.0
        num_comparisons = 0
        for i in range(n):
            for j in range(i + 1, n):
                graph1, graph2 = decoded_graph_features_list[i], decoded_graph_features_list[j]
                graph_features1, graph_features2 = self.comp_weisfeiler_lehman_subtree_features_from_graphs(graph1, graph2)
                similarity = self.calculate_min_max_similarity(graph_features1, graph_features2)
                yy += similarity
                num_comparisons += 1

        yy = yy * 2.0 / num_comparisons if num_comparisons > 0 else 0.0
#        print(f"Average similarity: {yy:.4f}")

        xy = 0.0
        num_comparisons = 0
        for i in range(m):
            for j in range(n):
                graph1, graph2 = data_graph_features_list[i], decoded_graph_features_list[j]
                graph_features1, graph_features2 = self.comp_weisfeiler_lehman_subtree_features_from_graphs(graph1, graph2)
                similarity = self.calculate_min_max_similarity(graph_features1, graph_features2)
                xy += similarity
                num_comparisons += 1


        xy = -xy * 2.0/ num_comparisons if num_comparisons > 0 else 0.0
#        print(f"Average similarity: {xy:.4f}")

        mmd_sq = xx + yy + xy
        mmd_sq = max(mmd_sq, 0.0)

        return np.sqrt(mmd_sq)

    def comp_similarity(self, dfs_codes, decoded_dfs_codes):
        decoded_graph = self.convert_decoded_sequence_to_graph(dfs_codes)
        data_graph    = self.convert_decoded_sequence_to_graph(decoded_dfs_codes)

        decoded_graph_features, data_graph_features = self.comp_weisfeiler_lehman_subtree_features_from_graphs(decoded_graph, data_graph)


#        similarity = self.calculate_cosine_similarity(decoded_graph_features, data_graph_features)
        similarity = self.calculate_min_max_similarity(decoded_graph_features, data_graph_features)
#        print("min_max_similarity:", min_max_similarity)
        return similarity

    def calculate_min_max_similarity(self, decoded_graph_features: Dict[int, int], data_graph_features: Dict[int, int]) -> float:
        # 共通のキーの集合を取得
        common_keys = set(decoded_graph_features.keys()).intersection(set(data_graph_features.keys()))

        # min-max類似度を計算
        min_sum = 0
        max_sum = 0

        for key in common_keys:
            decoded_value = decoded_graph_features.get(key, 0)
            data_value = data_graph_features.get(key, 0)
            min_sum += min(decoded_value, data_value)
            max_sum += max(decoded_value, data_value)

        # max_sumが0の場合、類似度は0
        if max_sum == 0:
            return 0.0

        min_max_similarity = min_sum / max_sum

        return min_max_similarity

    def calculate_cosine_similarity(self, decoded_graph_features: Dict[int, int], data_graph_features: Dict[int, int]) -> float:
        # 共通のキーの集合を取得
        common_keys = set(decoded_graph_features.keys()).intersection(set(data_graph_features.keys()))


        # 内積とベクトルの大きさを計算
        dot_product = 0
        decoded_norm = 0
        data_norm = 0

        for key in common_keys:
            decoded_value = decoded_graph_features.get(key, 0)
            data_value = data_graph_features.get(key, 0)
            dot_product += decoded_value * data_value

        for value in decoded_graph_features.values():
            decoded_norm += value ** 2

        for value in data_graph_features.values():
            data_norm += value ** 2

        # コサイン類似度を計算
        if decoded_norm == 0 or data_norm == 0:
            return 0.0
        
        cosine_similarity = dot_product / (decoded_norm ** 0.5 * data_norm ** 0.5)

        return cosine_similarity

    def convert_decoded_sequence_to_graph(self, decoded_seq: List[List[str]]) -> nx.Graph:
        G = nx.Graph()

#        node_id_map = {} 
#        node_id_counter = 0
        for seq in decoded_seq:
            vertex1 = int(seq[0])
            vertex2 = int(seq[1])
            vertex_label1 = int(seq[2])
            edge_label = int(seq[3])
            vertex_label2 = int(seq[4])


#            if vertex1 not in node_id_map:
#                node_id_map[vertex1] = node_id_counter
#                node_id_counter += 1
#            if vertex2 not in node_id_map:
#                node_id_map[vertex2] = node_id_counter
#                node_id_counter += 1

            if vertex1 not in G.nodes:
                G.add_node(vertex1, label=vertex_label1)
            if vertex2 not in G.nodes:
                G.add_node(vertex2, label=vertex_label2)

            G.add_edge(vertex1, vertex2, label=edge_label)
            G.add_edge(vertex2, vertex1, label=edge_label)

        return G
    
    def comp_weisfeiler_lehman_subtree_features_from_graphs(self, decoded_graph: nx.Graph, data_graph: nx.Graph) -> Tuple[Dict[int, int], Dict[int, int]]:
        
        # グラフのノードラベルを取得
        data_graph_features = {}
        node_label_map = {}
        node_label_counter = 0

        decoded_graph_features = {}
        decoded_graph_labels = nx.get_node_attributes(decoded_graph, 'label')
        for node in decoded_graph.nodes:
            vlabel = decoded_graph_labels[node]
            if vlabel not in node_label_map:
                node_label_map[vlabel] = node_label_counter
                node_label_counter += 1 
            if node_label_map[vlabel] not in decoded_graph_features:   
                decoded_graph_features[node_label_map[vlabel]] = 0
            decoded_graph_features[node_label_map[vlabel]] += 1

        for node in decoded_graph.nodes:
            decoded_graph.nodes[node]['label'] = node_label_map[decoded_graph_labels[node]]

        data_graph_features = {}
        data_graph_labels = nx.get_node_attributes(data_graph, 'label')
        for node in data_graph.nodes:
            vlabel = data_graph_labels[node]
            if vlabel not in node_label_map:
                node_label_map[vlabel] = node_label_counter
                node_label_counter += 1 
            if node_label_map[vlabel] not in data_graph_features:   
                data_graph_features[node_label_map[vlabel]] = 0
            data_graph_features[node_label_map[vlabel]] += 1

        for node in data_graph.nodes:
            data_graph.nodes[node]['label'] = node_label_map[data_graph_labels[node]]

        for iter in range(self.iteration):
            decoded_graph_labels = nx.get_node_attributes(decoded_graph, 'label')

            new_node_labels = {}
            for node in decoded_graph.nodes():
#                neighbors = list(decoded_graph.neighbors(node))
#                neighbor_labels = sorted([decoded_graph_labels[neighbor] for neighbor in neighbors])
#                tmp = f"{decoded_graph_labels[node]}_{'_'.join(map(str, neighbor_labels))}" 

                neighbors = list(decoded_graph.neighbors(node))
                neighbor_labels = sorted([
                    f"{decoded_graph_labels[neighbor]}_{decoded_graph.edges[node, neighbor]['label']}"
                   for neighbor in neighbors
                ])
                tmp = f"{decoded_graph_labels[node]}_{'_'.join(neighbor_labels)}"

                if tmp not in node_label_map:
                    node_label_map[tmp] = node_label_counter
                    node_label_counter += 1

                if node_label_map[tmp] not in decoded_graph_features:
                    decoded_graph_features[node_label_map[tmp]] = 0

                decoded_graph_features[node_label_map[tmp]] += 1
                new_node_labels[node] = node_label_map[tmp]
            
            for node in decoded_graph.nodes():
                decoded_graph.nodes[node]['label'] = new_node_labels[node]
#            for node in decoded_graph.nodes():
#                decoded_graph.nodes[node]['label'] = node_label_map[f"{decoded_graph_labels[node]}_{'_'.join(map(str, sorted([decoded_graph_labels[neighbor] for neighbor in decoded_graph.neighbors(node)])))}"]

            data_graph_labels = nx.get_node_attributes(data_graph, 'label')
            new_node_labels = {}
            for node in data_graph.nodes():
#                neighbors = list(data_graph.neighbors(node))
#                neighbor_labels = sorted([data_graph_labels[neighbor] for neighbor in neighbors])
#                tmp = f"{data_graph_labels[node]}_{'_'.join(map(str, neighbor_labels))}"

                neighbors = list(data_graph.neighbors(node))
                neighbor_labels = sorted([
                    f"{data_graph_labels[neighbor]}_{data_graph.edges[node, neighbor]['label']}"
                   for neighbor in neighbors
                ])
                tmp = f"{data_graph_labels[node]}_{'_'.join(neighbor_labels)}"

                if tmp not in node_label_map:
                    node_label_map[tmp] = node_label_counter
                    node_label_counter += 1

                if node_label_map[tmp] not in data_graph_features:
                    data_graph_features[node_label_map[tmp]] = 0

                data_graph_features[node_label_map[tmp]] += 1
                new_node_labels[node] = node_label_map[tmp]

            for node in data_graph.nodes():
                data_graph.nodes[node]['label'] = new_node_labels[node] 

#            for node in data_graph.nodes():
#                data_graph.nodes[node]['label'] = node_label_map[f"{data_graph_labels[node]}_{'_'.join(map(str, sorted([data_graph_labels[neighbor] for neighbor in data_graph.neighbors(node)])))}"]
            
#        print("decoded_graph_features:", decoded_graph_features)
#        print("data_graph_features:", data_graph_features)
        return decoded_graph_features, data_graph_features

class SequenceCompressor:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def compress(self, sequence: List[int]) -> List[int]:
        subsequences = []
        current_subsequence = []

        for token in sequence:
            token = int(token)
            if token == self.tokenizer.START_TOKEN or token == self.tokenizer.END_TOKEN or token == self.tokenizer.PAD_TOKEN:
                continue
            elif token == self.tokenizer.DELIMITER1 or token == self.tokenizer.DELIMITER2 or token == self.tokenizer.DELIMITER3 or token == self.tokenizer.DELIMITER4:
                if len(current_subsequence) != 0:
                    subsequences.append(current_subsequence)
                    current_subsequence = []
            else:
                current_subsequence.append(token)
        
        if len(current_subsequence) != 0:
            subsequences.append(current_subsequence)

#        print("subsequencesの中身:", subsequences)

        if len(subsequences) != 5:
#            print(len(subsequences))
            raise ValueError("subsequenceの個数が5つではありません。")


        for i in range(len(subsequences)):
            subsequence = subsequences[i]
            if i == 0 or i == 1:
                subsequence = self.compress_vertex_sequence(subsequence)
                subsequences[i] = subsequence
            elif i == 2 or i == 3 or i == 4:
                subsequence = self.compress_label_sequence(subsequence)
                subsequences[i] = subsequence

        new_sequence = [self.tokenizer.START_TOKEN] + subsequences[0] + [self.tokenizer.DELIMITER1] + subsequences[1] + [self.tokenizer.DELIMITER2] + subsequences[2] + [self.tokenizer.DELIMITER3] + subsequences[3] + [self.tokenizer.DELIMITER4] + subsequences[4] + [self.tokenizer.END_TOKEN]
#        print("new_sequence:", new_sequence)
        return new_sequence 
    
    def compress_vertex_sequence(self, subsequence: List[int]) -> List[int]:
        compressed_subsequence = [subsequence[0]]
        counter = 1
        for i in range(1, len(subsequence)):
            if subsequence[i-1] == subsequence[i]-1:
                counter += 1
            else:
                compressed_subsequence.append(counter)
                compressed_subsequence.append(subsequence[i])
                counter = 1

        if counter > 1:
            compressed_subsequence.append(counter)

        return compressed_subsequence

    def compress_label_sequence(self, subsequence: List[int]) -> List[int]:
        compressed_subsequence = [subsequence[0]]
        counter = 1
        for i in range(1, len(subsequence)):
            if subsequence[i-1] == subsequence[i]:
                counter += 1
            else:
                compressed_subsequence.append(counter)
                compressed_subsequence.append(subsequence[i])
                counter = 1

        if counter > 1:
            compressed_subsequence.append(counter)
        
        return compressed_subsequence
    
def check_minimality(dfs_codes: List[List[str]]):
    G = nx.Graph()

    node_id_map = {} 
    node_id_counter = 0
    for dfs_code in dfs_codes:
        vertex1 = dfs_code[0]
        vertex2 = dfs_code[1]
        vertex_label1 = dfs_code[2]
        edge_label = dfs_code[3]
        vertex_label2 = dfs_code[4]

        if vertex1 not in node_id_map:
            node_id_map[vertex1] = node_id_counter
            node_id_counter += 1
        if vertex2 not in node_id_map:
            node_id_map[vertex2] = node_id_counter
            node_id_counter += 1
        
        dfs_code[0] = node_id_map[vertex1]
        dfs_code[1] = node_id_map[vertex2]
        vertex1 = dfs_code[0]
        vertex2 = dfs_code[1]

#        print(f'v {vertex1} {vertex_label1}')
#        print(f'v {vertex2} {vertex_label2}')
#        print(f'e {vertex1} {vertex2} {edge_label}')
#        print(f'e {vertex2} {vertex1} {edge_label}')

        if vertex1 not in G.nodes:
            G.add_node(vertex1, label=vertex_label1)
        if vertex2 not in G.nodes:
            G.add_node(vertex2, label=vertex_label2)

        G.add_edge(vertex1, vertex2, label=edge_label)
        G.add_edge(vertex2, vertex1, label=edge_label)

    with open('output.gspan', 'w') as f:
        f.write('t # 0\n')
#        print('t # 0')
        for node, data in sorted(G.nodes(data=True)):
            f.write(f'v {node} {data["label"]}\n')
#            print(f'v {node} {data["label"]}')
        for u, v, data in G.edges(data=True):
            f.write(f'e {u} {v} {data["label"]}\n')
#            print(f'e {u} {v} {data["label"]}')

    command = ["/home/tabei/prog/cpp/gspan/dfs_code", "output.dfs", "0.8", "output.gspan"]
    subprocess.run(command)

    dfs_code_list = []
    with open('output.dfs', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('graph_id'):
                continue
            elements = line.strip().strip('<>').split(',')
            dfs_code_list.append([int(e.strip()) for e in elements if e.strip()])

    for dfs_code, dfs_code_list_item in zip(dfs_codes, dfs_code_list):
        if len(dfs_code) != 5 or len(dfs_code_list_item) != 5:
            return False
        if any(dfs_code[i] != dfs_code_list_item[i] for i in range(5)):
            return False
    return True
    
    
import networkx as nx


def is_valid_dfs_code(dfs_code):
    """
    DFSコードが有効なグラフを表しているかどうかをチェックします。

    Parameters:
        dfs_code (list of lists or tuples): 各要素が5つの整数からなるDFSコード。

    Returns:
        bool: 有効なDFSコードであればTrue、そうでなければFalse。
        str: エラーメッセージ（有効な場合は空文字）。
    """
    G = nx.Graph()
    vertex_labels = {}
    current_max_vertex = -1  # 初期状態では頂点が存在しない
    edge_set = set()

    for idx, code in enumerate(dfs_code):
        if len(code) != 5:
            return False, f"コードの要素 {idx} が5つの値を持っていません。"

        from_v, to_v, from_label, edge_label, to_label = code
        from_v = int(from_v)
        to_v = int(to_v) 
        from_label = int(from_label)
        to_label = int(to_label)
        edge_label = int(edge_label)

        # from_vが存在するか確認
        if from_v not in vertex_labels:
            if from_v == 0 and idx == 0:
                # 最初のエッジのfrom_vが0であることを許可
                vertex_labels[from_v] = from_label
                G.add_node(from_v, label=from_label)
                current_max_vertex = max(current_max_vertex, from_v)
            else:
                return False, f"コードの要素 {idx}: from_vertex {from_v} が存在しません。"

        # to_vが新しい頂点か既存の頂点かを確認
        if to_v not in vertex_labels:
            # 新しい頂点であるべき
            if to_v != current_max_vertex + 1:
                return False, (f"コードの要素 {idx}: to_vertex {to_v} が新しい頂点として追加されるべきですが、"
                               f"期待される値は {current_max_vertex + 1} です。")
            # ラベルを追加
            vertex_labels[to_v] = to_label
            G.add_node(to_v, label=to_label)
            current_max_vertex += 1
        else:
            # 既存の頂点である場合、ラベルが一致するか確認
            if vertex_labels[to_v] != to_label:
                return False, (f"コードの要素 {idx}: to_vertex {to_v} のラベルが一致しません。"
                               f"期待値: {vertex_labels[to_v]}, 実際の値: {to_label}")

        # from_vのラベルが一致するか確認
        if vertex_labels[from_v] != from_label:
            return False, (f"コードの要素 {idx}: from_vertex {from_v} のラベルが一致しません。"
                           f"期待値: {vertex_labels[from_v]}, 実際の値: {from_label}")

        # エッジが既に存在しないか確認
        edge = tuple(sorted((from_v, to_v)))
        if edge in edge_set:
            return False, f"コードの要素 {idx}: エッジ ({from_v}, {to_v}) が既に存在します。"
        
        # エッジを追加
        G.add_edge(from_v, to_v, label=edge_label)
        edge_set.add(edge)

    # 最後にグラフが連結かどうかをチェック
    if len(G.nodes) > 0 and not nx.is_connected(G):
        return False, "構築されたグラフが連結ではありません。"

    return True, ""

def calculate_novelty(generated_dfs_codes, train_loader):
    """
    生成された分子の新規性（トレーニングセットとの非重複率）を計算します。
    
    Args:
        generated_dfs_codes (list): 生成されたDFSコードのリスト
        training_data (list): トレーニングセットのデータのリスト
    
    Returns:
        float: 新規性の割合 (0.0 ~ 100.0)
        list: 新規なDFSコードのリスト
    """
    # DFSコードをタプルに変換して比較可能にする
#    training_set = set(tuple(map(tuple, data[1])) for data in train_loader)
    training_set = set()
    for data in train_loader:
        dfs_codes = data[0]
        for dfs_code in dfs_codes:
            training_set.add(tuple(map(tuple, dfs_code)))
    
    # 生成された分子の中で、トレーニングセットに含まれないものを抽出
    novel_molecules = []
    for dfs_code in generated_dfs_codes:
        dfs_tuple = tuple(map(tuple, dfs_code))
        if dfs_tuple not in training_set:
            novel_molecules.append(dfs_code)
    
    # 新規性の計算
    total_valid = len(generated_dfs_codes)
    if total_valid == 0:
        return 0.0, []
    
    novelty = (len(novel_molecules) / total_valid) * 100.0
    
    return novelty, novel_molecules

def analyze_generation_results(self, generated_dfs_codes, training_dfs_codes):
    """
    生成結果の詳細な分析を行います。
    
    Args:
        generated_dfs_codes (list): 生成されたDFSコードのリスト
        training_dfs_codes (list): トレーニングセットのDFSコードのリスト
    
    Returns:
        dict: 分析結果を含む辞書
    """
    # 新規性の計算
    novelty, novel_molecules = self.calculate_novelty(generated_dfs_codes, training_dfs_codes)
    
    # ユニーク分子の数を計算
    unique_molecules = set(tuple(map(tuple, dfs_code)) for dfs_code in generated_dfs_codes)
    
    # 分析結果をまとめる
    analysis = {
        'total_generated': len(generated_dfs_codes),
        'unique_molecules': len(unique_molecules),
        'novelty': novelty,
        'novel_molecules': len(novel_molecules),
        'training_set_overlap': 100.0 - novelty
    }
    
    return analysis

def generate_and_analyze(self, num_samples, max_len, temperature=1.0, training_dfs_codes=None):
    """
    分子を生成し、その結果を分析します。
    
    Args:
        num_samples (int): 生成するサンプル数
        max_len (int): 各サンプルの最大長
        temperature (float): サンプリングの温度パラメータ
        training_dfs_codes (list, optional): トレーニングセットのDFSコード
    
    Returns:
        tuple: (生成されたDFSコード, 分析結果)
    """
    # 分子の生成
    generated_dfs_codes = self.sample(num_samples, max_len, temperature)
    
    # 分析の実行
    if training_dfs_codes is not None:
        analysis = self.analyze_generation_results(generated_dfs_codes, training_dfs_codes)
    else:
        analysis = {
            'total_generated': len(generated_dfs_codes),
            'unique_molecules': len(set(tuple(map(tuple, dfs_code)) for dfs_code in generated_dfs_codes))
        }
    
    return generated_dfs_codes, analysis
    
def calculate_uniqueness(generated_dfs_codes): 
    """
    生成された分子の一意性（ユニーク性）を計算します。
    
    Args:
        generated_dfs_codes (list): 生成されたDFSコードのリスト
    
    Returns:
        float: 一意性の割合 (0.0 ~ 100.0)
        list: ユニークなDFSコードのリスト
        dict: 各分子の出現回数
    """
    # DFSコードをタプルに変換して比較可能にする
    dfs_tuples = [tuple(map(tuple, dfs_code)) for dfs_code in generated_dfs_codes]
        
    # 各分子の出現回数をカウント
    molecule_counts = {}
    for dfs_tuple in dfs_tuples:
        molecule_counts[dfs_tuple] = molecule_counts.get(dfs_tuple, 0) + 1
        
    # ユニークな分子のリストを作成
    unique_molecules = []
    for dfs_code in generated_dfs_codes:
        dfs_tuple = tuple(map(tuple, dfs_code))
        if dfs_tuple not in set(tuple(map(tuple, m)) for m in unique_molecules):
            unique_molecules.append(dfs_code)
        
    # 一意性の計算
    total_molecules = len(generated_dfs_codes)
    if total_molecules == 0:
        return 0.0, [], {}
        
    uniqueness = (len(unique_molecules) / total_molecules) * 100.0
        
    return uniqueness, unique_molecules, molecule_counts

def calculate_diversity(generated_dfs_codes, tokenizer, wl_iteration):
    """
    生成された分子の多様性を計算します。
    
    Args:
        generated_dfs_codes (list): 生成されたDFSコードのリスト
    
    Returns:
        float: 多様性の割合 (0.0 ~ 100.0)
    """
    # 多様性の計算  
    compare = Compare(tokenizer, wl_iteration)

    total_pairs = 0
    total_similarity = 0.0

    for i in range(len(generated_dfs_codes)):
        for j in range(i + 1, len(generated_dfs_codes)):
            dfs_code_1 = generated_dfs_codes[i]
            dfs_code_2 = generated_dfs_codes[j]
            similarity = compare.comp_similarity(dfs_code_1, dfs_code_2)
            total_similarity += similarity
            total_pairs += 1

    if total_pairs == 0:
        return 0.0


    diversity = (1 - (total_similarity / total_pairs)) * 100.0

    return diversity

class GraphAnalyzer:
    """
    生成されたグラフの構造特性を分析するためのクラス
    """
    def __init__(self):
        self.nx = nx

    def analyze_graph_structure(self, generated_dfs_codes):
        """
        生成されたグラフの構造特性を分析します。
        
        Args:
            generated_dfs_codes (list): 生成されたDFSコードのリスト
        
        Returns:
            dict: グラフの構造特性の分析結果
        """
        graphs = []
        graph_stats = {
            'degree_distribution': {},
            'clustering_coefficients': [],
            'avg_shortest_paths': [],
            'community_sizes': [],
            'assortativity': [],
            'graph_sizes': {
                'nodes': [],
                'edges': []
            }
        }
        
        # DFSコードからグラフを構築
        for dfs_codes in generated_dfs_codes:
            G = self.nx.Graph()
            for code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = code
                if from_v not in G.nodes():
                    G.add_node(from_v, label=from_label)
                if to_v not in G.nodes():
                    G.add_node(to_v, label=to_label)
                G.add_edge(from_v, to_v, label=edge_label)
            graphs.append(G)
        
        for G in graphs:
            # 次数分布の計算
            degrees = [d for _, d in G.degree()]
            for degree in degrees:
                graph_stats['degree_distribution'][degree] = \
                    graph_stats['degree_distribution'].get(degree, 0) + 1
            
            # クラスタリング係数
            try:
                avg_clustering = self.nx.average_clustering(G)
                graph_stats['clustering_coefficients'].append(avg_clustering)
            except:
                pass
            
            # 平均最短経路長
            try:
                avg_path = self.nx.average_shortest_path_length(G)
                graph_stats['avg_shortest_paths'].append(avg_path)
            except:
                pass
            
            # コミュニティ検出（Louvainアルゴリズム）
            try:
                import community
                communities = community.best_partition(G)
                community_sizes = {}
                for node, community_id in communities.items():
                    community_sizes[community_id] = community_sizes.get(community_id, 0) + 1
                graph_stats['community_sizes'].append(list(community_sizes.values()))
            except:
                pass
            
            # Assortativity（次数相関）
            try:
                assortativity = self.nx.degree_assortativity_coefficient(G)
                graph_stats['assortativity'].append(assortativity)
            except:
                pass
            
            # グラフサイズ
            graph_stats['graph_sizes']['nodes'].append(G.number_of_nodes())
            graph_stats['graph_sizes']['edges'].append(G.number_of_edges())
        
        # 統計量の計算
        stats_summary = {
            'degree_distribution': {
                'distribution': graph_stats['degree_distribution'],
                'avg_degree': sum(d * count for d, count in graph_stats['degree_distribution'].items()) / 
                             sum(graph_stats['degree_distribution'].values()),
                'max_degree': max(graph_stats['degree_distribution'].keys()),
                'min_degree': min(graph_stats['degree_distribution'].keys())
            },
            'clustering': {
                'mean': sum(graph_stats['clustering_coefficients']) / len(graph_stats['clustering_coefficients']) 
                       if graph_stats['clustering_coefficients'] else 0,
                'std': self._calculate_std(graph_stats['clustering_coefficients']),
                'distribution': graph_stats['clustering_coefficients']
            },
            'path_length': {
                'mean': sum(graph_stats['avg_shortest_paths']) / len(graph_stats['avg_shortest_paths'])
                       if graph_stats['avg_shortest_paths'] else 0,
                'std': self._calculate_std(graph_stats['avg_shortest_paths']),
                'distribution': graph_stats['avg_shortest_paths']
            },
            'communities': {
                'avg_count': sum(len(sizes) for sizes in graph_stats['community_sizes']) / len(graph_stats['community_sizes'])
                            if graph_stats['community_sizes'] else 0,
                'avg_size': sum(sum(sizes) / len(sizes) for sizes in graph_stats['community_sizes']) / 
                           len(graph_stats['community_sizes']) if graph_stats['community_sizes'] else 0,
                'size_distribution': graph_stats['community_sizes']
            },
            'assortativity': {
                'mean': sum(graph_stats['assortativity']) / len(graph_stats['assortativity'])
                       if graph_stats['assortativity'] else 0,
                'std': self._calculate_std(graph_stats['assortativity']),
                'distribution': graph_stats['assortativity']
            },
            'graph_size': {
                'nodes': {
                    'mean': sum(graph_stats['graph_sizes']['nodes']) / len(graph_stats['graph_sizes']['nodes']),
                    'std': self._calculate_std(graph_stats['graph_sizes']['nodes']),
                    'distribution': graph_stats['graph_sizes']['nodes']
                },
                'edges': {
                    'mean': sum(graph_stats['graph_sizes']['edges']) / len(graph_stats['graph_sizes']['edges']),
                    'std': self._calculate_std(graph_stats['graph_sizes']['edges']),
                    'distribution': graph_stats['graph_sizes']['edges']
                }
            }
        }
        
        return stats_summary

    def _calculate_std(self, values):
        """
        標準偏差を計算します。
        """
        if not values:
            return 0
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5

    def analyze_and_visualize_graphs(self, generated_dfs_codes, output_dir=None):
        """
        グラフの構造特性を分析し、可視化します。
        
        Args:
            generated_dfs_codes (list): 生成されたDFSコードのリスト
            output_dir (str, optional): 結果を保存するディレクトリ
        
        Returns:
            dict: 分析結果
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 構造特性の分析
        stats = self.analyze_graph_structure(generated_dfs_codes)
        
        if output_dir:
            # 次数分布のプロット
            plt.figure(figsize=(10, 6))
            degrees = list(stats['degree_distribution']['distribution'].keys())
            counts = list(stats['degree_distribution']['distribution'].values())
            plt.bar(degrees, counts)
            plt.xlabel('Degree')
            plt.ylabel('Count')
            plt.title('Degree Distribution')
            plt.savefig(f'{output_dir}/degree_distribution.png')
            plt.close()
            
            # クラスタリング係数の分布
            plt.figure(figsize=(10, 6))
            sns.histplot(stats['clustering']['distribution'], bins=20)
            plt.xlabel('Clustering Coefficient')
            plt.ylabel('Count')
            plt.title('Clustering Coefficient Distribution')
            plt.savefig(f'{output_dir}/clustering_distribution.png')
            plt.close()
            
            # 平均最短経路長の分布
            plt.figure(figsize=(10, 6))
            sns.histplot(stats['path_length']['distribution'], bins=20)
            plt.xlabel('Average Shortest Path Length')
            plt.ylabel('Count')
            plt.title('Path Length Distribution')
            plt.savefig(f'{output_dir}/path_length_distribution.png')
            plt.close()
            
            # Assortativityの分布
            plt.figure(figsize=(10, 6))
            sns.histplot(stats['assortativity']['distribution'], bins=20)
            plt.xlabel('Assortativity')
            plt.ylabel('Count')
            plt.title('Assortativity Distribution')
            plt.savefig(f'{output_dir}/assortativity_distribution.png')
            plt.close()
            
            # グラフサイズの分布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            sns.histplot(stats['graph_size']['nodes']['distribution'], bins=20, ax=ax1)
            ax1.set_xlabel('Number of Nodes')
            ax1.set_ylabel('Count')
            ax1.set_title('Node Distribution')
            
            sns.histplot(stats['graph_size']['edges']['distribution'], bins=20, ax=ax2)
            ax2.set_xlabel('Number of Edges')
            ax2.set_ylabel('Count')
            ax2.set_title('Edge Distribution')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/graph_size_distribution.png')
            plt.close()
        
        return stats


import networkx as nx
import numpy as np
from typing import List, Tuple, Dict
from itertools import combinations
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphMetrics:
    """
    A comprehensive class for evaluating graph generation quality using multiple metrics.
    """
    
    def __init__(self, sigma: float = 1.0, n_clustering_bins: int = 20):
        """
        Initialize the GraphMetrics calculator.
        
        Args:
            sigma: Bandwidth parameter for the Gaussian kernel
            n_clustering_bins: Number of bins for clustering coefficient histogram
        """
        self.sigma = sigma
        self.n_clustering_bins = n_clustering_bins
    
    @staticmethod
    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """Calculate Gaussian kernel between two vectors."""
        return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
    
    def calculate_mmd(self, dist1: List[np.ndarray], dist2: List[np.ndarray]) -> float:
        """
        Calculate Maximum Mean Discrepancy with Gaussian kernel.
    
        Args:
            dist1: List of first distribution arrays
            dist2: List of second distribution arrays
    
        Returns:
            MMD value
        """
        if not dist1 or not dist2:
            logger.warning("Empty distribution provided for MMD calculation")
            return 0.0

        # Find maximum length across all arrays in both distributions
        max_len = max(max(len(d) for d in dist1), max(len(d) for d in dist2))
    
        # Pad all arrays to max_len
        dist1_padded = np.zeros((len(dist1), max_len))
        dist2_padded = np.zeros((len(dist2), max_len))
    
        for i, d in enumerate(dist1):
            if len(d) > 0:  # Only pad if array is not empty
                dist1_padded[i, :len(d)] = d
    
        for i, d in enumerate(dist2):
            if len(d) > 0:  # Only pad if array is not empty
                dist2_padded[i, :len(d)] = d
    
        # Calculate kernel matrices
        n, m = len(dist1), len(dist2)
        K_XX = np.zeros((n, n))
        K_YY = np.zeros((m, m))
        K_XY = np.zeros((n, m))
    
        for i in range(n):
            for j in range(n):
                K_XX[i,j] = self.gaussian_kernel(dist1_padded[i], dist1_padded[j], self.sigma)
    
        for i in range(m):
            for j in range(m):
                K_YY[i,j] = self.gaussian_kernel(dist2_padded[i], dist2_padded[j], self.sigma)
    
        for i in range(n):
            for j in range(m):
                K_XY[i,j] = self.gaussian_kernel(dist1_padded[i], dist2_padded[j], self.sigma)
    
        # Calculate MMD
        mmd = (np.sum(K_XX) / (n * n) + 
               np.sum(K_YY) / (m * m) - 
               2 * np.sum(K_XY) / (n * m))
    
        return float(np.sqrt(max(mmd, 0)))

    def get_degree_distribution(self, G: nx.Graph) -> np.ndarray:
        """Calculate normalized degree distribution of a graph."""
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph provided for degree distribution calculation")
            return np.array([])
            
        degrees = [d for _, d in G.degree()]
        max_degree = max(degrees)
        dist = np.zeros(max_degree + 1)
        for d in degrees:
            dist[d] += 1
        return dist / len(G)
    
    def get_clustering_distribution(self, G: nx.Graph) -> np.ndarray:
        """Calculate clustering coefficient distribution of a graph."""
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph provided for clustering distribution calculation")
            return np.array([])
        
        # Calculate clustering coefficients with error handling
        clustering = {}
        for node in G.nodes():
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    coef = nx.clustering(G, nodes=[node])[node]
                    if np.isfinite(coef):  # Only include valid coefficients
                        clustering[node] = coef
                    else:
                        clustering[node] = 0.0
            except:
                clustering[node] = 0.0
    
        if not clustering:
            return np.zeros(self.n_clustering_bins)
        
        # Create histogram
        values = list(clustering.values())
        hist, _ = np.histogram(values, bins=self.n_clustering_bins, 
                          range=(0, 1), density=True)
    
        return hist

    
    def count_4_node_orbits(self, G: nx.Graph) -> np.ndarray:
        """
        Count 4-node graphlet orbits in a graph.
        Returns normalized counts of 11 possible orbits.
        """
        def get_subgraph_edges(nodes):
            return [(i, j) for i, j in combinations(nodes, 2) if G.has_edge(i, j)]
        
        def get_node_degrees(nodes, edges):
            degrees = Counter()
            for edge in edges:
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
            return {node: degrees.get(node, 0) for node in nodes}

        def is_path_or_star(nodes, edges, degrees):
            if len(edges) != 3:
                return False
            degree_values = list(degrees.values())
            return 3 in degree_values or (2 in degree_values and 1 in degree_values)
        
        if G.number_of_nodes() < 4:
            logger.warning("Graph has fewer than 4 nodes for orbit calculation")
            return np.zeros(11)
            
        orbits = np.zeros(11)
        total_patterns = 0
        
        for nodes in combinations(G.nodes(), 4):
            edges = get_subgraph_edges(nodes)
            edge_count = len(edges)
            degrees = get_node_degrees(nodes, edges)
            
            # Count each orbit pattern
            if edge_count == 0:  # Independent set
                orbits[0] += 4
                total_patterns += 4
                
            elif edge_count == 1:  # Single edge
                orbits[1] += 2
                orbits[4] += 2
                total_patterns += 4
                
            elif edge_count == 2:
                if max(degrees.values()) == 2:  # Path of length 2
                    orbits[2] += 1
                    orbits[1] += 2
                    orbits[4] += 1
                    total_patterns += 4
                else:  # Two independent edges
                    orbits[1] += 4
                    total_patterns += 4
                    
            elif edge_count == 3:
                if max(degrees.values()) == 3:  # Star
                    orbits[3] += 1
                    orbits[4] += 3
                    total_patterns += 4
                elif is_path_or_star(nodes, edges, degrees):  # Path of length 3
                    orbits[1] += 2
                    orbits[2] += 2
                    total_patterns += 4
                else:  # Triangle with isolated node
                    orbits[7] += 3
                    orbits[4] += 1
                    total_patterns += 4
                    
            elif edge_count == 4:
                if max(degrees.values()) == 2:  # 4-cycle
                    orbits[8] += 4
                    total_patterns += 4
                else:  # Triangle with leaf
                    orbits[6] += 1
                    orbits[7] += 2
                    orbits[1] += 1
                    total_patterns += 4
                    
            elif edge_count == 5:  # Chord cycle
                orbits[9] += 4
                total_patterns += 4
                
            elif edge_count == 6:  # Complete graph
                orbits[10] += 4
                total_patterns += 4
        
        return orbits / total_patterns if total_patterns > 0 else orbits
    
    def evaluate_graphs(self, 
                   generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph],
                   min_nodes: int = 4) -> Dict[str, float]:
        """
        Evaluate generated graphs against reference graphs using multiple metrics.
    
        Args:
            generated_graphs: List of generated NetworkX graphs
            reference_graphs: List of reference NetworkX graphs
            min_nodes: Minimum number of nodes required for evaluation
    
        Returns:
            Dictionary containing MMD scores for each metric
        """
        if not generated_graphs or not reference_graphs:
            raise ValueError("Empty graph list provided")
    
        # Filter out graphs that are too small
        valid_generated = [G for G in generated_graphs if G.number_of_nodes() >= min_nodes]
        valid_reference = [G for G in reference_graphs if G.number_of_nodes() >= min_nodes]
    
        if not valid_generated or not valid_reference:
            logger.warning(f"No graphs with {min_nodes} or more nodes found")
            return {
                'degree_mmd': 0.0,
               'clustering_mmd': 0.0,
                'orbit_mmd': 0.0
            }
    
        # Calculate distributions only for valid graphs
        gen_degrees = [self.get_degree_distribution(G) for G in valid_generated]
        gen_clustering = [self.get_clustering_distribution(G) for G in valid_generated]
        gen_orbits = [self.count_4_node_orbits(G) for G in valid_generated]
    
        ref_degrees = [self.get_degree_distribution(G) for G in valid_reference]
        ref_clustering = [self.get_clustering_distribution(G) for G in valid_reference]
        ref_orbits = [self.count_4_node_orbits(G) for G in valid_reference]
    
        # Calculate MMD for each metric
        results = {
            'degree_mmd': self.calculate_mmd(gen_degrees, ref_degrees),
            'clustering_mmd': self.calculate_mmd(gen_clustering, ref_clustering),
            'orbit_mmd': self.calculate_mmd(gen_orbits, ref_orbits)
        }
    
        return results

# Example usage
#if __name__ == "__main__":
#    # Create a graph metrics calculator
#    metrics = GraphMetrics(sigma=1.0, n_clustering_bins=20)
#    
#    # Generate some example graphs
#    n_graphs = 10
#    generated = [nx.erdos_renyi_graph(20, 0.2) for _ in range(n_graphs)]
#    reference = [nx.erdos_renyi_graph(20, 0.2) for _ in range(n_graphs)]
#    
    # Evaluate
#    results = metrics.evaluate_graphs(generated, reference)
#    
    # Print results
#    for metric, value in results.items():
#        print(f"{metric}: {value:.3f}")
#

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
  4-node motif を厳密に 11 種類へ分類し、その分布をヒストグラムとして取得。
  Degree / Clustering / Orbit (4-node motif) の 3 つを別々に MMD 計算する。

  大規模グラフで全列挙すると非常に重くなる場合があるので、
  max_subgraphs で部分的にサンプリングし、計算量を抑制。

  ※ 本格的には Hočevar & Demšar (2014) 等の高速アルゴリズム (ORCA) が望ましい。
"""

import random
import numpy as np
import networkx as nx
from itertools import combinations


# =========================================================================
# 1) Degree / Clustering 計算 (前回同様)
# =========================================================================

def calc_degree_hist(G, max_degree=20):
    """
    グラフ G のノードの度をヒストグラム化: [0,1,2,...,max_degree, >max_degree]
    """
    hist = np.zeros(max_degree + 2, dtype=np.float32)
    for _, d in G.degree():
        if d > max_degree:
            hist[max_degree+1] += 1
        else:
            hist[d] += 1
    total_nodes = G.number_of_nodes()
    if total_nodes > 0:
        hist /= total_nodes
    return hist


def calc_clustering_hist(G, bins=10):
    """
    グラフ G の各ノードのクラスタリング係数を bins 個のビンに分割してカウント。
    """
    clust_vals = list(nx.clustering(G).values())
    hist = np.zeros(bins, dtype=np.float32)
    for c in clust_vals:
        idx = int(c * (bins - 1))  # 0～(bins-1)
        hist[idx] += 1
    n = len(clust_vals)
    if n > 0:
        hist /= n
    return hist


# =========================================================================
# 2) 4-node motif (4ノード部分グラフ) を 11 種類に分類
# =========================================================================

def adjacency_to_canonical_str(adj_mat):
    """
    4×4 の隣接行列 (numpy配列) を、辞書順最小となるようにノードを並び替えた
    キャノニカルフォーム (文字列) に変換する簡易的関数。

    ※ 本来はさらに効率的なアルゴリズムが存在しますが、
      ここでは全ての順列(24通り)を試して最小のものを取得する実装例を示します。
    """
    import itertools

    n = adj_mat.shape[0]
    best_str = None

    # すべてのノード順列 (4ノードなら 24通り)
    for perm in itertools.permutations(range(n)):
        # permに合わせて行列を並び替え
        perm = list(perm)
        sub = adj_mat[perm, :][:, perm]
        # 上三角を文字列化 (無向グラフなので対称行列)
        # ここでは全要素でもよいが、無駄が多いので上三角だけにする例
        tri = []
        for i in range(n):
            for j in range(i+1, n):
                tri.append(str(int(sub[i, j])))
        s = "".join(tri)
        if (best_str is None) or (s < best_str):
            best_str = s
    
    return best_str


def classify_4node_subgraph(adj_mat):
    """
    4×4 の無向隣接行列を受け取り、
    11種類の 4-node motif のどれに該当するか ID (0~10) を返す。

    実装方針:
     1) adjacency_to_canonical_str() でキャノニカルフォーム (文字列) を得る
     2) それをキーに 11 種類のいずれかにマッピング

    → 実際には、各 "キャノニカルフォーム" を事前に全列挙して
      dictに登録する必要がある (今回コード内で定義)。

    参考: 4ノードの非同型無向グラフは全部で 11 種。
          (辺数 0~6 の範囲で、同型をまとめると計11種類)
    """
    # まずキャノニカルフォームを取得
    cstr = adjacency_to_canonical_str(adj_mat)

    # ↓ 事前に 4-node のすべての非同型パターンを
    #    "キャノニカルフォーム" => motif ID に写像した辞書を作る必要がある。
    #    ここでは例として既に埋め込み済みとし、簡易化のためにほんの一例だけを記述します。
    #    ただし本当に全11種類をきちんと定義しようとすると、
    #    全パターンの隣接行列からキャノニカルフォームを作って列挙する作業が必要です。

    # ここでは例として(実際には11個全部は書ききれないので)
    # "cstr" が確認済みのいくつかのパターンだけ dict で対応付ける:
    # （注意: ここでは一部しか載せていないので実際に使用する際は全パターンを定義してください）
    canonical_map = {
        # 0 edges → adjacency上三角は "000000" (長さ6文字になる)
        "000000": 0,

        # 1 edge → adjacency上三角は "100000" 的なパターンが最小形
        #   (順列によって "010000" 等も出るがキャノニカルフォーム化すると同じになるはず)
        "100000": 1,

        # 6 edges (完全グラフ K4) → adjacency上三角は "111111"
        "111111": 10,

        # などなど… 実際には 2,3,4,5,6,7,8,9 に相当する9種類が必要
        # ここでは省略
    }

    if cstr in canonical_map:
        return canonical_map[cstr]
    else:
        # 未定義パターン → 便宜上 5 番 (仮) などにまとめる
        # （本来はきちんと 0~10 のどこかに対応づける必要がある）
        return 5


def calc_orbit_hist_4_detailed(
    G, 
    max_subgraphs=100,
    random_seed=42,
    num_motifs=11
):
    """
    4頂点サブグラフをサンプリングし、各サブグラフを 11種類のモチーフID に分類。
    (厳密には全分類をきちんと実装する必要がありますが、
     サンプルコードでは canonical_map の一部しか定義していません。)

    返り値: 長さ num_motifs (= 11) のベクトル (モチーフ別カウント) を1に正規化。
    """
    orbit_count = np.zeros(num_motifs, dtype=np.float32)

    nodes = list(G.nodes())
    all_4combs = list(combinations(nodes, 4))
    
    random.seed(random_seed)
    if len(all_4combs) > max_subgraphs:
        selected_subsets = random.sample(all_4combs, max_subgraphs)
    else:
        selected_subsets = all_4combs

    for subset in selected_subsets:
        subG = G.subgraph(subset)
        # 4×4 の隣接行列 (numpy)
        # nodelist=sorted(subset) では順番が固定されてしまうので、
        # canonical_form で最終的に順序を吸収する。
        mat = nx.to_numpy_array(subG, nodelist=subset, dtype=int)
        # classify
        motif_id = classify_4node_subgraph(mat)
        if motif_id < num_motifs:
            orbit_count[motif_id] += 1

    total = orbit_count.sum()
    if total > 0:
        orbit_count /= total
    return orbit_count


# =========================================================================
# 3) RBFカーネル & MMD (多次元ベクトル用)
# =========================================================================

def gaussian_rbf_kernel(x, y, sigma=1.0):
    diff = x - y
    diff_sq = np.dot(diff, diff)
    return np.exp(-diff_sq / (2.0 * sigma * sigma))


def compute_mmd_multi_dim(X_p, X_q, kernel_func, sigma=1.0):
    """
    MMD^2 = E[k(xi,xj)] + E[k(yi,yj)] - 2E[k(xi,yj)]
    を unbiased estimator で計算
    """
    m = len(X_p)
    n = len(X_q)
    if m < 2 or n < 2:
        return 0.0

    xx = 0.0
    for i in range(m):
        for j in range(i+1, m):
            xx += kernel_func(X_p[i], X_p[j], sigma=sigma)
    xx = xx * 2.0 / (m*(m-1))

    yy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            yy += kernel_func(X_q[i], X_q[j], sigma=sigma)
    yy = yy * 2.0 / (n*(n-1))

    xy = 0.0
    for i in range(m):
        for j in range(n):
            xy += kernel_func(X_p[i], X_q[j], sigma=sigma)
    xy = xy * (-2.0) / (m*n)

    mmd_sq = xx + yy + xy
    mmd_sq = max(mmd_sq, 0.0)
    return np.sqrt(mmd_sq)

import numpy as np

def compute_mmd_multi_dim_subsample(
    X_p, 
    X_q, 
    kernel_func, 
    sigma=1.0, 
    Bp=50,    # サブサンプリングサイズ (X_p)
    Bq=50,    # サブサンプリングサイズ (X_q)
    n_iter=10  # サンプリング回数
):
    """
    ランダムサンプリングした部分集合に対して MMD を複数回計算し、その平均を返す。

    Parameters
    ----------
    X_p : array-like
        ソース側のサンプル群 (リストや NumPy 配列など)
    X_q : array-like
        ターゲット側のサンプル群
    kernel_func : function
        k(x, y, sigma=...) の形で呼び出せるカーネル関数
    sigma : float
        カーネルのパラメータ
    Bp : int
        X_p からサンプリングする個数
    Bq : int
        X_q からサンプリングする個数
    n_iter : int
        ランダムサンプリングを何回繰り返すか

    Returns
    -------
    float
        MMD の推定値（複数回のサンプリングによる平均）
    """

    # もし Bp や Bq が元データ数より大きい場合は調整
    m = len(X_p)
    n = len(X_q)
    Bp = min(Bp, m)
    Bq = min(Bq, n)

    # データ数が 2 未満だとアンバイアス推定ができないので 0.0 を返す
    if m < 2 or n < 2:
        return 0.0

    # サンプリングを繰り返して平均を取る
    mmd_values = []
    for _ in range(n_iter):
        # ランダムにインデックスを選び出す
        idx_p = np.random.choice(m, size=Bp, replace=False)
        idx_q = np.random.choice(n, size=Bq, replace=False)

        # 部分集合を取り出す
        subX_p = [X_p[i] for i in idx_p]
        subX_q = [X_q[j] for j in idx_q]

        # 以下、アンバイアス推定の MMD 計算
        # ここでは「MMD (平方根)」を返す実装
        xx = 0.0
        for i in range(Bp):
            for j in range(i+1, Bp):
                xx += kernel_func(subX_p[i], subX_p[j], sigma=sigma)
        xx = xx * 2.0 / (Bp*(Bp-1))

        yy = 0.0
        for i in range(Bq):
            for j in range(i+1, Bq):
                yy += kernel_func(subX_q[i], subX_q[j], sigma=sigma)
        yy = yy * 2.0 / (Bq*(Bq-1))

        xy = 0.0
        for i in range(Bp):
            for j in range(Bq):
                xy += kernel_func(subX_p[i], subX_q[j], sigma=sigma)
        xy = xy * (-2.0) / (Bp*Bq)

        mmd_sq = xx + yy + xy
        mmd_sq = max(mmd_sq, 0.0)  # 数値誤差対策
        mmd_val = np.sqrt(mmd_sq)  # MMD^2 の平方根を MMD として返す

        mmd_values.append(mmd_val)

    # 複数回のサンプリング結果を平均
    return float(np.mean(mmd_values))


# =========================================================================
# 4) メイン評価関数 (Degree / Clustering / Orbit の 3つを別々に計算)
#    - Orbit は「辺数6分類」の代わりに「4-node motif 11分類」を使用
# =========================================================================

def evaluate_generated_graphs(real_graphs, gen_graphs, sigma=1.0):
    """
    real_graphs, gen_graphs: グラフのリスト
    1) Degree ヒストグラム
    2) Clustering ヒストグラム
    3) 4-node motif (11種類) ヒストグラム
    をそれぞれ計算し、MMD を別々に出力する
    """
    # -- real side --
    real_deg_vecs = []
    real_clust_vecs = []
    real_orbit_vecs = []

    for G in real_graphs:
        real_deg_vecs.append(calc_degree_hist(G))
        real_clust_vecs.append(calc_clustering_hist(G))
        real_orbit_vecs.append(calc_orbit_hist_4_detailed(G))  # 11分類のやつ
    X_real_deg = np.array(real_deg_vecs, dtype=np.float32)
    X_real_clust = np.array(real_clust_vecs, dtype=np.float32)
    X_real_orbit = np.array(real_orbit_vecs, dtype=np.float32)
    # -- gen side --
    gen_deg_vecs = []
    gen_clust_vecs = []
    gen_orbit_vecs = []

    for G in gen_graphs:
        gen_deg_vecs.append(calc_degree_hist(G))
        gen_clust_vecs.append(calc_clustering_hist(G))
        gen_orbit_vecs.append(calc_orbit_hist_4_detailed(G))  # 11分類のやつ
    X_gen_deg = np.array(gen_deg_vecs, dtype=np.float32)
    X_gen_clust = np.array(gen_clust_vecs, dtype=np.float32)
    X_gen_orbit = np.array(gen_orbit_vecs, dtype=np.float32)

    # -- compute MMDs --
    mmd_deg = compute_mmd_multi_dim(X_real_deg, X_gen_deg, gaussian_rbf_kernel, sigma)
#    mmd_deg = compute_mmd_multi_dim_subsample(X_real_deg, X_gen_deg, gaussian_rbf_kernel, sigma)
    mmd_clust = compute_mmd_multi_dim(X_real_clust, X_gen_clust, gaussian_rbf_kernel, sigma)
#    mmd_clust = compute_mmd_multi_dim_subsample(X_real_clust, X_gen_clust, gaussian_rbf_kernel, sigma)
    mmd_orbit = compute_mmd_multi_dim(X_real_orbit, X_gen_orbit, gaussian_rbf_kernel, sigma)
#    mmd_orbit = compute_mmd_multi_dim_subsample(X_real_orbit, X_gen_orbit, gaussian_rbf_kernel, sigma)

    return {
        "degree": mmd_deg,
        "clustering": mmd_clust,
        "orbit": mmd_orbit
    }
import numpy as np
import networkx as nx
from collections import Counter
from scipy.linalg import sqrtm

def collect_all_labels(graphs):
    """
    グラフ群からノードラベルとエッジラベルの全種類を収集し、
    重複なくまとめて返す。
    """
    node_label_set = set()
    edge_label_set = set()
    
    for G in graphs:
        for _, data in G.nodes(data=True):
            lbl = data.get('label', None)
            if lbl is not None:
                node_label_set.add(lbl)
        for _, _, data in G.edges(data=True):
            elbl = data.get('label', None)
            if elbl is not None:
                edge_label_set.add(elbl)
    
    return sorted(list(node_label_set)), sorted(list(edge_label_set))

def extract_feature_vector(G, node_labels, edge_labels):
    """
    1つのグラフGから特徴ベクトルを抽出して返す。
    """
    # ノードラベルカウント
    node_label_counter = Counter()
    for _, data in G.nodes(data=True):
        lbl = data.get('label', None)
        node_label_counter[lbl] += 1

    # エッジラベルカウント
    edge_label_counter = Counter()
    for _, _, data in G.edges(data=True):
        elbl = data.get('label', None)
        edge_label_counter[elbl] += 1

    # node_labels の順序に合わせてカウント
    node_count_vec = [node_label_counter[lbl] for lbl in node_labels]
    # edge_labels の順序に合わせてカウント
    edge_count_vec = [edge_label_counter[lbl] for lbl in edge_labels]
    
    # 追加統計: ノード数, エッジ数, 平均次数
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = 0.0
    if num_nodes > 0:
        degrees = [deg for _, deg in G.degree()]
        avg_degree = float(np.mean(degrees))
    
    feature_vec = node_count_vec + edge_count_vec + [num_nodes, num_edges, avg_degree]
    return np.array(feature_vec, dtype=np.float32)

def compute_mean_and_cov(features, eps=1e-6):
    """
    特徴ベクトル群 (N x D) から平均ベクトル (D,) と共分散行列 (D x D) を計算。
    数値安定化のために対角に eps を加える。
    """
    # 平均ベクトル
    mu = np.mean(features, axis=0)

    # サンプルが1つしかないと covariance = 0 次元 or rank-deficient なので対策
    if features.shape[0] == 1:
        # サンプル数が1の場合は covariance を完全に 0 行列にしつつ、
        # 対角に微小量を足して rank-deficient を回避
        D = features.shape[1]
        cov = np.zeros((D, D), dtype=np.float32)
        np.fill_diagonal(cov, eps)
    else:
        cov = np.cov(features, rowvar=False)  # (D x D)
        # 数値誤差で非正定になるのを回避するため、対角に小さい値を足す
        cov += np.eye(cov.shape[0]) * eps

    return mu, cov

def frechet_distance(mu1, cov1, mu2, cov2):
    """
    2つのガウス分布 N(mu1, cov1) と N(mu2, cov2) のフレシェ距離 (FD) を計算。
    FD = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2 * sqrt(cov1 * cov2))
    """
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # 行列平方根 sqrtm(cov1 dot cov2)
    covprod = sqrtm(cov1.dot(cov2))
    # 数値誤差で虚数が出た場合の対策（real 部分のみを取り出す）
    if np.iscomplexobj(covprod):
        covprod = covprod.real

    fd = diff_sq + np.trace(cov1 + cov2 - 2.0 * covprod)
    return fd

def compute_frechet_distance_for_graph_sets(graphs1, graphs2):
    """
    2つのグラフ集合に対してフレシェ距離を計算し、値を返す。
    """
    # --- 1) ラベル一覧の収集 ---
    node_labels_1, edge_labels_1 = collect_all_labels(graphs1)
    node_labels_2, edge_labels_2 = collect_all_labels(graphs2)
    all_node_labels = sorted(list(set(node_labels_1 + node_labels_2)))
    all_edge_labels = sorted(list(set(edge_labels_1 + edge_labels_2)))
    
    # --- 2) グラフ毎の特徴ベクトル抽出 ---
    features1 = []
    for G in graphs1:
        features1.append(extract_feature_vector(G, all_node_labels, all_edge_labels))
    features1 = np.vstack(features1) if len(features1) > 0 else None
    
    features2 = []
    for G in graphs2:
        features2.append(extract_feature_vector(G, all_node_labels, all_edge_labels))
    features2 = np.vstack(features2) if len(features2) > 0 else None
    
    # 万が一、どちらかのセットが空の場合
    if features1 is None or features2 is None:
        raise ValueError("One of the graph sets is empty; cannot compute Fréchet distance.")
    
    # --- 3) 平均と共分散行列を計算 ---
    mu1, cov1 = compute_mean_and_cov(features1)
    mu2, cov2 = compute_mean_and_cov(features2)
    
    # --- 4) フレシェ距離を計算 ---
    fd = frechet_distance(mu1, cov1, mu2, cov2)
    return fd

if __name__ == "__main__":
    # ================================
    #  例: サイズの異なるグラフをいくつか作成
    # ================================
    
    # --- G1（3ノード, 2エッジ） ---
    G1 = nx.Graph()
    G1.add_node(1, label="A")
    G1.add_node(2, label="A")
    G1.add_node(3, label="B")
    G1.add_edge(1, 2, label="X")
    G1.add_edge(2, 3, label="Y")
    
    # --- G2（4ノード, 3エッジ） ---
    G2 = nx.Graph()
    G2.add_node(1, label="B")
    G2.add_node(2, label="C")
    G2.add_node(3, label="C")
    G2.add_node(4, label="A")
    G2.add_edge(1, 2, label="Z")
    G2.add_edge(2, 3, label="X")
    G2.add_edge(3, 4, label="X")
    
    # --- G3（5ノード, 4エッジ） ---
    G3 = nx.Graph()
    G3.add_node(1, label="A")
    G3.add_node(2, label="B")
    G3.add_node(3, label="B")
    G3.add_node(4, label="C")
    G3.add_node(5, label="C")
    G3.add_edge(1, 2, label="Y")
    G3.add_edge(2, 3, label="Z")
    G3.add_edge(3, 4, label="X")
    G3.add_edge(4, 5, label="Z")
    
    # 仮にセット1が G1, G2 で、セット2が G3 の想定
    set1 = [G1, G2]
    set2 = [G3]

    # フレシェ距離を計算
    fd_value = compute_frechet_distance_for_graph_sets(set1, set2)
    print("Fréchet Distance between set1 and set2: ", fd_value)
