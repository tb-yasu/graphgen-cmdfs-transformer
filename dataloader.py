import csv
import json
from typing import List, Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence
from util import Pair
#from util import RLBWT
from util import SequenceCompressor
from util import Tokenizer

class DFSDataset(Dataset):
    def __init__(self, tokenizer, dfs_file, disable_label: bool = False):
        self.tokenizer = tokenizer
        
        dfs_codes_list = self.load_dfs_codes_list(dfs_file)

        if disable_label == False:
            dfs_sequences = self.tokenizer.encode_dfs_codes_list_to_sequences2(dfs_codes_list)
        else:
            dfs_codes_list_nolabels = []
            for dfs_codes in dfs_codes_list:
                dfs_codes_nolabels = []
                for dfs_code in dfs_codes:
                    from_v, to_v, from_label, edge_label, to_label = map(int, dfs_code)
                    dfs_codes_nolabels.append([from_v, to_v, 1, 1, 1])
                dfs_codes_list_nolabels.append(dfs_codes_nolabels)
            dfs_codes_list = dfs_codes_list_nolabels
            dfs_sequences = self.tokenizer.encode_dfs_codes_list_to_sequences_no_label(dfs_codes_list_nolabels)

        len_original_dfs_codes_list = 0
        for dfs_codes in dfs_codes_list:
            len_original_dfs_codes_list += len(dfs_codes) * 5
        print(f"average len_original_dfs_codes_list: {len_original_dfs_codes_list / len(dfs_codes_list)}")

        len_dfs_sequences = 0
        for dfs_sequence in dfs_sequences:
            len_dfs_sequences += len(dfs_sequence)
        print(f"average len_dfs_sequences: {len_dfs_sequences / len(dfs_sequences)}")

#        decoded_dfs_codes_list = self.tokenizer.decode_dfs_codes_sequences_to_dfs_codes_list(data_dfs_sequences)

#        for i, dfs_sequence in enumerate(dfs_sequences):
#            print(f"dfs_sequence {i}:", dfs_sequence)

        self.data = self.merge_data(dfs_codes_list, dfs_sequences)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        指定したidx番目のrowのdfs_codesとdfs_sequenceを返す

        Args:
            idx (int): 取得したいデータのインデックス

        Returns:
            Tuple[List[int], List[int]]: dfs_codesとdfs_sequenceのタプル
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.data)}")

        dfs_codes = self.data[idx]['dfs_codes']
        dfs_sequence = self.data[idx]['dfs_sequence']

        return dfs_codes, dfs_sequence

    def merge_data(self, dfs_codes_list, dfs_sequences):

        if len(dfs_codes_list) != len(dfs_sequences):
            print("len(dfs_codes_list), len(dfs_sequences):", len(dfs_codes_list), len(dfs_sequences))
            raise ValueError("dfs_codes_listとdfs_sequencesの要素数が一致しません。")

        merged_data = []
        for i in range(len(dfs_codes_list)):  
            dfs_codes = dfs_codes_list[i]
            dfs_sequence = dfs_sequences[i]

            merged_entry = {}
            merged_entry['dfs_codes'] = dfs_codes
            merged_entry['dfs_sequence'] = dfs_sequence

            merged_data.append(merged_entry)
        return merged_data

    def load_dfs_codes_list(
        self, 
        file_path: str
        ) -> List[List[List[int]]]:
        """
        指定されたファイルから複数のDFSコードを読み込み、各グラフごとに
        頂点、頂点ラベル、エッジラベルに一意のIDを割り振ったシーケンスをリストとして保存します。
        また、各シーケンスの最初と最後に特殊トークンを追加し、必要に応じてパディングを行います。
        さらに、デコード時に使用する逆マッピングを生成します。

        Args:
            file_path (str): DFSコードが記述されたファイルのパス。
            max_length (int, optional): シーケンスの最大長。指定しない場合は全てのシーケンスの長さを最大長に揃える。

        Returns:
            list of list: 各グラフのDFSコードを一意のIDで直列に連結し、特殊トークンとパディングを適用したリストのリスト。
        """

        dfs_codes_list = []
        current_dfs_codes = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('graph_id:'):
                    if len(current_dfs_codes) != 0:
                        # シーケンスの開始と終了に特殊トークンを追加
                        dfs_codes_list.append(current_dfs_codes)
                        current_dfs_codes = []
                    continue

                # '<' と '>' を取り除き、カンマで分割
                line_content = line.strip('<>').strip()
                parts = [int(part.strip()) for part in line_content.split(',')]

                if len(parts) != 5:
                    continue  # 項目数が正しくない行をスキップ

                current_dfs_codes.append(parts)

            # 最後のシーケンスを追加
            if len(current_dfs_codes) != 0:
                dfs_codes_list.append(current_dfs_codes)

        return dfs_codes_list


    def decode_sequence(self, sequence: List[int]) -> List[str]:
        """
        シーケンスをデコードして元のDFSコードを再構築します。

        Args:
            sequence (list): デコード対象のシーケンス。

        Returns:
            list: デコードされたDFSコードのリスト。
        """
        dfs_code = []
        i = 0
        while i < len(sequence):
            token = sequence[i]
            if token == self.tokenizer.START_TOKEN:  # START_TOKEN
                i += 1
                continue
            elif token == self.tokenizer.END_TOKEN:  # END_TOKEN
                break
            elif token == self.tokenizer.PAD_TOKEN:  # PAD_TOKEN
                i += 1
                continue
            else:
                # 頂点1 ID, 頂点2 ID, 頂点ラベル1 ID, エッジラベル ID, 頂点ラベル2 ID
                if i + 4 >= len(sequence):
                    print("シーケンスが不完全です。")
                    break
                vertex1_id = sequence[i]
                vertex2_id = sequence[i+1]
                vertex1_label_id = sequence[i+2]
                edge_label_id = sequence[i+3]
                vertex2_label_id = sequence[i+4]
                i += 5
                
#                vertex1 = self.tokenizer.id_to_vertex1.get(vertex1_id, "未知の頂点")
#                vertex2 = self.tokenizer.id_to_vertex2.get(vertex2_id, "未知の頂点")
#                vertex1_label = self.tokenizer.id_to_vertex_label1.get(vertex1_label_id, "未知のラベル")
#                edge_label = self.tokenizer.id_to_edge_label.get(edge_label_id, "未知のエッジラベル")
#                vertex2_label = self.tokenizer.id_to_vertex_label2.get(vertex2_label_id, "未知のラベル")

                vertex1 = self.tokenizer.id_to_vertex.get(vertex1_id, "未知の頂点") 
                vertex2 = self.tokenizer.id_to_vertex.get(vertex2_id, "未知の頂点")
                vertex1_label = self.tokenizer.id_to_vertex_label.get(vertex1_label_id, "未知のラベル")
                edge_label = self.tokenizer.id_to_edge_label.get(edge_label_id, "未知のエッジラベル")
                vertex2_label = self.tokenizer.id_to_vertex_label.get(vertex2_label_id, "未知のラベル")

                dfs_code.append(f"<{vertex1}, {vertex2}, {vertex1_label}, {edge_label}, {vertex2_label}>")

        return dfs_code

    def save_mappings(self, filepath: str):
        """
        マッピングとシーケンスをJSONファイルとして保存します。

        Args:
            filepath (str): 保存先のファイルパス。
        """
        data = {
            'vertex_map': self.tokenizer.vertex_map,
            'vertex_label_map': self.tokenizer.vertex_label_map,
            'edge_label_map': self.tokenizer.edge_label_map,
            'id_to_vertex': self.tokenizer.id_to_vertex,
            'id_to_vertex_label': self.tokenizer.id_to_vertex_label,
            'id_to_edge_label': self.tokenizer.id_to_edge_label,
            'all_sequences': self.all_sequences,
            'START_TOKEN': self.tokenizer.START_TOKEN,
            'END_TOKEN': self.tokenizer.END_TOKEN,
            'PAD_TOKEN': self.tokenizer.PAD_TOKEN
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"マッピングとシーケンスを {filepath} に保存しました。")

    def load_mappings(self, filepath: str):
        """
        JSONファイルからマッピングとシーケンスを読み込みます。

        Args:
            filepath (str): 読み込み元のファイルパス。
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tokenizer.vertex_map = {k: v for k, v in data['vertex_map'].items()}
        self.tokenizer.vertex_label_map = {k: v for k, v in data['vertex_label_map'].items()}
        self.tokenizer.edge_label_map = {k: v for k, v in data['edge_label_map'].items()}
        self.tokenizer.id_to_vertex = {int(k): v for k, v in data['id_to_vertex'].items()}
        self.tokenizer.id_to_vertex_label = {int(k): v for k, v in data['id_to_vertex_label'].items()}
        self.tokenizer.id_to_edge_label = {int(k): v for k, v in data['id_to_edge_label'].items()}
        self.all_sequences = data['all_sequences']
        self.tokenizer.START_TOKEN = data.get('START_TOKEN', 0)
        self.tokenizer.END_TOKEN = data.get('END_TOKEN', 1)
        self.tokenizer.PAD_TOKEN = data.get('PAD_TOKEN', 2)

        print(f"マッピングとシーケンスを {filepath} から読み込みました。")

    def get_all_sequences(self) -> List[List[int]]:
        """
        すべてのシーケンスを取得します。

        Returns:
            list of list: 全シーケンスのリスト。
        """
        return self.all_sequences

class DFSDataLoader(DataLoader):
    def __init__(self, tokenizer, dfs_file: str, batch_size: int = 32, train_rate: float = 0.8, label_usage: bool = True):
        self.dfs_file = dfs_file
        self.batch_size = batch_size
        self.train_rate = train_rate

        self.dataset = DFSDataset(tokenizer, dfs_file, label_usage)
            
#            rlbwt = RLBWT(self.dataset.data[i]['dfs_sequence'][1:-1])
#            rlbwt.construct_bwt()
#            rlbwt.rle_encode()
#            print(rlbwt.rlbwt_list)
#            exit(1)

        
        self.PAD_TOKEN = self.dataset.tokenizer.PAD_TOKEN

        super().__init__(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        dfs_codes     = [item[0] for item in batch]
        dfs_sequences = [item[1] for item in batch]

        # DFSシーケンスをtorch.tensorに変換し、padding
        dfs_tensors = [torch.tensor(seq, dtype=torch.long) for seq in dfs_sequences]
        dfs_tensors_padded = pad_sequence(dfs_tensors, batch_first=True, padding_value=self.dataset.tokenizer.PAD_TOKEN)

        return dfs_codes, dfs_tensors_padded
    
    def get_dataloader(self):
        train_size = int(self.train_rate * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=0)
        
        return train_loader, test_loader
