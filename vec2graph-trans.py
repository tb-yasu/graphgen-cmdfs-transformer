import torch
import torch.nn as nn
from util import Tokenizer
from util import get_device
import math
import torch.nn.functional as F
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class TransformerGraphTransformerGenerator(nn.Module):
    def __init__(self, tokenizer, emb_size, hidden_size, num_layers, nhead=8):

        super().__init__()
        self.embedding = nn.Embedding(
            num_embedding=tokenizer.vocab_size(),         
            embedding_dim=emb_size, 
            padding_idx=tokenizer.PAD_TOKEN       
        )
        self.pos_encoder = PositionalEncoding(d_model=emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model=emb_size, out_features=tokenizer.vocab_size())

    def forward(self, src, src_mask=None):
        # src: (batch, seq_len)
        x = self.embedding(src)          # (batch, seq_len, d_model)
        x = self.pos_encoder(x)          # (batch, seq_len, d_model)
        x = x.transpose(0, 1)            # (seq_len, batch, d_model) for Transformer
        x = self.transformer(x, src_mask)
        x = x.transpose(0, 1)            # (batch, seq_len, d_model)
        return self.output(x)            # (batch, seq_len, vocab_size)


class GraphGenerator(nn.Module):
    def __init__(self, tokenizer, emb_size, hidden_size, num_layers, dropout=0.2):
        """
        tokenizer: contains SMILES string tokens
        emb_size: embedding dimension
        hidden_size: number of hidden neurons of RNN
        num_layers: number of layers
        dropout: dropout probability
        """
        super(GraphGenerator, self).__init__()

        self.tokenizer = tokenizer
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


        # Embedding層   
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size(), 
            embedding_dim=self.emb_size, 
            padding_idx=self.tokenizer.PAD_TOKEN
        )

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=self.emb_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True, 
            bidirectional=False
        )
        
        # 全結合層
        self.fc = nn.Linear(self.hidden_size, self.tokenizer.vocab_size())

        # LogSoftmax層
        self.log_softmax = nn.LogSoftmax(dim=1)

        # パラメータの初期化
        self.init_params()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def init_hidden(self, batch_size):
        num_directions = 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return h0, c0

    def create_position_encoding(self, period, emb_size):
        position_encoding = torch.zeros(period, emb_size)
        for pos in range(period):
            for i in range(0, emb_size, 2):
                position_encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / emb_size)))
                if i + 1 < emb_size:
                    position_encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / emb_size)))
        return position_encoding

    def forward(self, decoder_inputs):
        """
        decoder_inputs: [batch_size, max_len] -> [batch_size, max_len, emb_size]
        returns:
            pred: [batch_size * max_len, vocab_size]
        """
        self.lstm.flatten_parameters()

        h0, c0 = self.init_hidden(decoder_inputs.size(0))
        
        embedded = self.embedding(decoder_inputs)  # [batch_size, max_len, emb_size]

        # LSTM層の出力
        output, _ = self.lstm(embedded, (h0, c0))  # [batch_size, max_len, hidden_size]

        # 全結合層を通して各タイムステップでの出力を得る
        logits = self.fc(output.contiguous().view(-1, self.hidden_size))  # [batch_size * max_len, vocab_size]
        
        # LogSoftmax層を通して予測を得る
        pred = self.log_softmax(logits)  # [batch_size * max_len, vocab_size]

        return pred
    
    def step(
        self, 
        decoder_input, 
        h, 
        c
    ):
        """
        Compute the output per time step

        decoder_input: [batch_size, 1]
        latent_vector: [batch_size, latent_size]
        h: hidden state [num_layers, batch_size, hidden_size]
        c: cell state [num_layers, batch_size, hidden_size]

        returns:
            pred: [batch_size, vocab_size]
            h: [batch_size, hidden_size]
            c: [batch_size, hidden_size]
        """


        self.lstm.flatten_parameters()
        
        embedded = self.embedding(decoder_input)  # [batch_size, 1, emb_size

        output, (h, c) = self.lstm(embedded, (h, c))  # [batch_size, 1, hidden_size]

        logits = self.fc(output.squeeze(1))

        pred = self.log_softmax(logits)  # [batch_size, vocab_size]
        
        return pred, h, c


    def apply_mask(self,output, indices):
        mask = torch.zeros_like(output)
        mask[:, indices] = 1
        return output.masked_fill(~mask.bool(), float('-inf'))

    def apply_mask_batch(self, output, indices_batch):
        """
        output: [batch_size, vocab_size]
        indices_batch: list of lists, each containing the indices to be masked for each batch element
        """
        mask = torch.zeros_like(output)
        for batch_idx, indices in enumerate(indices_batch):
            mask[batch_idx, indices] = 1
        return output.masked_fill(~mask.bool(), float('-inf'))


    def search_vertex_label(self, dfs_codes, v) -> int:
        for code in dfs_codes:
            from_v, to_v, from_label, edge_label, to_label = code
            if  v == from_v:
                return from_label
            elif v == to_v:
                return to_label

        return "<UNK>"

    def decode(self, max_len):
        """
            単一のサンプルを生成します: [[max_len]]
        
            Args:
                max_len (int): 
                latent_vector (torch.Tensor): 潜在ベクトル [latent_size]
        
            Returns:
                pred_dfs_codes (torch.Tensor): 生成されたdfs_codes [[max_len]]
        """

        batch_size = 1

        dfs_codes = []
        used_vertex_ids = set()  # これまでに出現したvertex_idを追跡

        h, c = self.init_hidden(batch_size)

        # 初期入力トークンをSTART_TOKENに設定
        x = torch.ones(batch_size, 1, dtype=torch.long) * self.tokenizer.START_TOKEN
        x = x.to(get_device())

        prev_verteces = set()
        max_vertex_id = 0
        prev_to_v = 0
        for i in range(max_len):
            # 1ステップごとに出力を生成
            if i == 0:
                from_v = 0
                to_v   = 1
                range_from_label_first = self.tokenizer.range_from_vertex_label_id.first
                range_from_label_second = self.tokenizer.range_from_vertex_label_id.second
                indices = list(range(range_from_label_first, range_from_label_second + 1))
                output, h, c = self.step(x, h, c)
                output = self.apply_mask(output, indices)
                output = self.log_softmax(output)
                x1 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
                from_label = self.tokenizer.id_to_vertex_label[x1.item()]

                range_edge_label_first = self.tokenizer.range_edge_label_id.first
                range_edge_label_second = self.tokenizer.range_edge_label_id.second
                indices = list(range(range_edge_label_first, range_edge_label_second + 1))
                output, h, c = self.step(x1, h, c)
                output = self.apply_mask(output, indices)
                output = torch.log_softmax(output, dim=1)
                x2 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
                edge_label = self.tokenizer.id_to_edge_label[x2.item()]

                range_to_label_first = self.tokenizer.range_vertex_label_id.first
                range_to_label_second = self.tokenizer.range_vertex_label_id.second
                indices = list(range(range_to_label_first, range_to_label_second + 1))
                output, h, c = self.step(x2, h, c)
                output = self.apply_mask(output, indices)
                output = self.log_softmax(output)
                x3 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
                to_label = self.tokenizer.id_to_vertex_label[x3.item()]

                dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])

                prev_to_v = to_v
                max_vertex_id = max(max_vertex_id, from_v)
                max_vertex_id = max(max_vertex_id, to_v)
                prev_verteces.add(from_v)
                prev_verteces.add(to_v)
                x = x3
            else:
                end_token = self.tokenizer.END_TOKEN
                range_edge_label_first = self.tokenizer.range_edge_label.first
                range_edge_label_second = self.tokenizer.range_edge_label.second
                indices = list(map(self.tokenizer.vertex_id_map.get, prev_verteces)) + [end_token] + list(range(range_edge_label_first, range_edge_label_second + 1))
                output, h, c = self.step(x, h, c)
                output = self.apply_mask(output, indices)
                output = self.log_softmax(output)
                x1 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]

                if self.tokenizer.END_TOKEN == x1:
                    break
                    
                if self.tokenizer.range_vertex_id.first <= x1.item() <= self.tokenizer.range_vertex_id.second:
                    range_edge_label_first = self.tokenizer.range_edge_label.first
                    range_edge_label_second = self.tokenizer.range_edge_label.second

                    indices = list(map(self.tokenizer.vertex_id_map.get, prev_verteces)) + list(range(range_edge_label_first, range_edge_label_second + 1))
                    output, h, c = self.step(x1, h, c)
                    output = self.apply_mask(output, indices)
                    output = self.log_softmax(output)
                    x2 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]

                    if self.tokenizer.range_to_vertex_id.first <= x2.item() <= self.tokenizer.range_to_vertex_id.second:
                        range_edge_label_first  = self.tokenizer.range_edge_label_id.first
                        range_edge_label_second = self.tokenizer.range_edge_label_id.second
                        indices = list(range(range_edge_label_first, range_edge_label_second + 1))
                        output, h, c = self.step(x2, h, c)
                        output = self.apply_mask(output, indices)
                        output = self.log_softmax(output)
                        x3 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]

                        from_v = self.tokenizer.id_to_vertex[x1.item()]
                        to_v   = self.tokenizer.id_to_vertex[x2.item()]
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        to_label   = self.search_vertex_label(dfs_codes, to_v)
                        edge_label = self.tokenizer.id_to_edge_label[x3.item()]
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        prev_to_v = to_v
                        max_vertex_id = max(max_vertex_id, from_v)
                        max_vertex_id = max(max_vertex_id, to_v)
                        x = x3
                    elif self.tokenizer.max_to >= max_vertex_id + 1:
                        range_to_label_first = self.tokenizer.range_to_vertex_label_id.first
                        range_to_label_second = self.tokenizer.range_to_vertex_label_id.second
                        indices = list(range(range_to_label_first, range_to_label_second + 1))
                        output, h, c = self.step(x2, h, c)
                        output = self.apply_mask(output, indices)
                        output = self.log_softmax(output)
                        x3 = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]

                        from_v = self.tokenizer.id_to_vertex[x1.item()]
                        to_v   = max_vertex_id + 1
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        edge_label = self.tokenizer.id_to_edge_label[x2.item()]
                        to_label   = self.tokenizer.id_to_vertex_label[x3.item()]
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        prev_to_v = to_v
                        max_vertex_id = max(max_vertex_id, from_v)
                        max_vertex_id = max(max_vertex_id, to_v)
                        prev_verteces.add(to_v)
                        x = x3
                elif self.tokenizer.max_to >= max_vertex_id + 1:
                    range_to_label_first  = self.tokenizer.range_to_vertex_label_id.first
                    range_to_label_second = self.tokenizer.range_to_vertex_label_id.second
                    indices = list(range(range_to_label_first, range_to_label_second + 1))
                    output, h, c = self.step(x1, h, c)
                    output = self.apply_mask(output, indices)
                    output = torch.log_softmax(output, dim=1)
                    x2 = torch.multinomial(torch.exp(output), 1)
                    from_v = prev_to_v
                    to_v   = max_vertex_id + 1
                    from_label = self.search_vertex_label(dfs_codes, from_v)
                    edge_label = self.tokenizer.id_to_edge_label[x1.item()]
                    to_label   = self.tokenizer.id_to_to_vertex_label[x2.item()]
                    dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                    prev_to_v = to_v
                    max_vertex_id = max(max_vertex_id, from_v)
                    max_vertex_id = max(max_vertex_id, to_v)
                    prev_verteces.add(to_v)
                    x = x2

        return dfs_codes


    def sample(self, num_samples, max_len, temperature=1.0):
        """
        複数のグラフサンプルを生成します。

        Args:
            num_samples (int): 生成するサンプル数
            max_len (int): 各サンプルの最大長
            temperature (float): サンプリングの温度パラメータ。
                               高いと多様性が増し、低いと決定的になります。

        Returns:
            list: 生成されたDFSコードのリスト。各要素���[from_v, to_v, from_label, edge_label, to_label]のリスト
        """
        self.eval()  # 評価モードに設定
        generated_samples = []
        
        i = 0
        for _ in range(num_samples):
            i += 1
            try:
                # 単一サンプルの生成
                dfs_codes = self.decode_with_temperature(max_len, temperature)
                generated_samples.append(dfs_codes)
            except Exception as e:
                print(f"サンプル生成中にエラーが発生: {str(e)}")
                continue
                
        return generated_samples

    def sample_no_label(self, num_samples, max_len, temperature=1.0):
        self.eval()  # 評価モードに設定
        generated_samples = []
        
        i = 0
        for _ in range(num_samples):
            i += 1
            try:
                # 単一サンプルの生成
                dfs_codes = self.decode_with_temperature_nolabels(max_len, temperature)
                generated_samples.append(dfs_codes)
            except Exception as e:
                print(f"サンプル生成中にエラーが発生: {str(e)}")
                continue
                
        return generated_samples

    def decode_with_temperature(self, max_len, temperature=1.0):
        """
        温度パラメータを考慮して単一のサンプルを生成します。

        Args:
            max_len (int): 生成する系列の最大長
            temperature (float): サンプリングの温度パラメータ

        Returns:
            list: 生成されたDFSコード
        """
        batch_size = 1
        dfs_codes = []
        used_vertex_ids = set()
        
        h, c = self.init_hidden(batch_size)
        
        # 初期入力トークン
        x = torch.ones(batch_size, 1, dtype=torch.long) * self.tokenizer.START_TOKEN
        x = x.to(get_device())
        
        prev_verteces = set()
        max_vertex_id = 0
        prev_to_v = 0
        
        def apply_temperature(logits):
            if temperature == 0:  # Greedy sampling
                return F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.size(-1))
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                return probs
        
        for i in range(max_len):
            # First token generation
            if i == 0:
                from_v = 0
                to_v   = 1
                
                # Generate from_label
                range_from_label_first = self.tokenizer.range_vertex_label.first
                range_from_label_second = self.tokenizer.range_vertex_label.second
                indices = list(range(range_from_label_first, range_from_label_second + 1))
                output, h, c = self.step(x, h, c)
                output = self.apply_mask(output, indices)
                probs = apply_temperature(output)
                x1 = torch.multinomial(probs, 1)
                from_label = self.tokenizer.id_to_vertex_label[x1.item()]
                # Generate edge_label
                range_edge_label_first = self.tokenizer.range_edge_label.first
                range_edge_label_second = self.tokenizer.range_edge_label.second
                indices = list(range(range_edge_label_first, range_edge_label_second + 1))
                output, h, c = self.step(x1, h, c)
                output = self.apply_mask(output, indices)
                probs = apply_temperature(output)
                x2 = torch.multinomial(probs, 1)
                edge_label = self.tokenizer.id_to_edge_label[x2.item()]
                # Generate to_label
                range_vertex_label_first = self.tokenizer.range_vertex_label.first
                range_vertex_label_second = self.tokenizer.range_vertex_label.second
                indices = list(range(range_vertex_label_first, range_vertex_label_second + 1))
                output, h, c = self.step(x2, h, c)
                output = self.apply_mask(output, indices)
                probs = apply_temperature(output)
                x3 = torch.multinomial(probs, 1)
                to_label = self.tokenizer.id_to_vertex_label[x3.item()]
                dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                prev_to_v = to_v
                max_vertex_id = max(max_vertex_id, to_v)
                prev_verteces.update([from_v, to_v])
                x = x3
            else:
#                print("2")
                end_token = self.tokenizer.END_TOKEN
                range_edge_label_first  = self.tokenizer.range_edge_label.first
                range_edge_label_second = self.tokenizer.range_edge_label.second
                indices = (
                    list(map(self.tokenizer.vertex_id_map.get, prev_verteces)) + 
                    [end_token] + 
                    list(range(range_edge_label_first, range_edge_label_second + 1))
                )
                
                output, h, c = self.step(x, h, c)
                output = self.apply_mask(output, indices)
                probs = apply_temperature(output)
                x1 = torch.multinomial(probs, 1)
                
                if x1.item() == self.tokenizer.END_TOKEN:
                    break
                
                if self.tokenizer.range_vertex_id.first <= x1.item() <= self.tokenizer.range_vertex_id.second:
#                    print("3")
                    # Continue generating the graph structure with temperature sampling
                    # (Similar logic as before but with temperature-based sampling)
                    indices = (
                        list(map(self.tokenizer.vertex_id_map.get, prev_verteces)) + 
                        list(range(range_edge_label_first, range_edge_label_second + 1))
                    )
                    output, h, c = self.step(x1, h, c)
                    output = self.apply_mask(output, indices)
                    probs = apply_temperature(output)
                    x2 = torch.multinomial(probs, 1)
                    
                    if self.tokenizer.range_vertex_id.first <= x2.item() <= self.tokenizer.range_vertex_id.second:
                        indices = list(range(range_edge_label_first, range_edge_label_second + 1))
                        output, h, c = self.step(x2, h, c)
                        output = self.apply_mask(output, indices)
                        probs = apply_temperature(output)
                        x3 = torch.multinomial(probs, 1)
                        
                        from_v = self.tokenizer.id_to_vertex_id[x1.item()]
                        to_v = self.tokenizer.id_to_vertex_id[x2.item()]
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        to_label = self.search_vertex_label(dfs_codes, to_v)
                        edge_label = self.tokenizer.id_to_edge_label[x3.item()]
                        
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        prev_to_v = to_v
                        max_vertex_id = max(max_vertex_id, to_v)
                        x = x3
                        
                    elif self.tokenizer.max_to >= max_vertex_id + 1:
                        range_vertex_label_first = self.tokenizer.range_vertex_label.first
                        range_vertex_label_second = self.tokenizer.range_vertex_label.second
                        indices = list(range(range_vertex_label_first, range_vertex_label_second + 1))
                        
                        output, h, c = self.step(x2, h, c)
                        output = self.apply_mask(output, indices)
                        probs = apply_temperature(output)
                        x3 = torch.multinomial(probs, 1)
                        
                        from_v = self.tokenizer.id_to_vertex_id[x1.item()]
                        to_v = max_vertex_id + 1
                        from_label = self.search_vertex_label(dfs_codes, from_v)
                        edge_label = self.tokenizer.id_to_edge_label[x2.item()]
                        to_label = self.tokenizer.id_to_vertex_label[x3.item()]
                        
                        dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                        prev_to_v = to_v
                        max_vertex_id = max(max_vertex_id, to_v)
                        prev_verteces.add(to_v)
                        x = x3
                
                elif self.tokenizer.max_to >= max_vertex_id + 1:
#                    print("4")
                    range_vertex_label_first = self.tokenizer.range_vertex_label.first
                    range_vertex_label_second = self.tokenizer.range_vertex_label.second
                    indices = list(range(range_vertex_label_first, range_vertex_label_second + 1))
                    
                    output, h, c = self.step(x1, h, c)
                    output = self.apply_mask(output, indices)
                    probs = apply_temperature(output)
                    x2 = torch.multinomial(probs, 1)
                    
                    from_v = prev_to_v
                    to_v = max_vertex_id + 1
                    from_label = self.search_vertex_label(dfs_codes, from_v)
                    edge_label = self.tokenizer.id_to_edge_label[x1.item()]
                    to_label = self.tokenizer.id_to_vertex_label[x2.item()]
                    
                    dfs_codes.append([from_v, to_v, from_label, edge_label, to_label])
                    prev_to_v = to_v
                    max_vertex_id = max(max_vertex_id, to_v)
                    prev_verteces.add(to_v)
                    x = x2
        
        return dfs_codes

    def decode_with_temperature_nolabels(self, max_len, temperature=1.0):
        """
        温度パラメータを考慮して単一のサンプルを生成します。

        Args:
            max_len (int): 生成する系列の最大長
            temperature (float): サンプリングの温度パラメータ

        Returns:
            list: 生成されたDFSコード
        """

        batch_size = 1
        dfs_codes = []
        used_vertex_ids = set()
        
        h, c = self.init_hidden(batch_size)
        
        # 初期入力トークン
        x = torch.ones(batch_size, 1, dtype=torch.long) * self.tokenizer.START_TOKEN
        x = x.to(get_device())
        
        prev_verteces = set()
#        dfs_codes.append([0, 1, 1, 1, 1])
#        prev_verteces.add(0)
#        prev_verteces.add(1)
#        prev_to_v = 1
#        max_vertex_id = 1

        def apply_temperature(logits):
            if temperature == 0:  # Greedy sampling
                return F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.size(-1))
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                return probs

        end_token = self.tokenizer.END_TOKEN

        indices = [0]
        output, h, c = self.step(x, h, c)
        output = self.apply_mask(output, indices)
        probs = apply_temperature(output)
        x1 = torch.multinomial(probs, 1)
        max_vertex_id = 0
        prev_to_v = 0
        prev_verteces.add(0)

#        indices = (list(map(self.tokenizer.vertex_id_map.get, prev_verteces)) + [1] + [end_token])
        indices = [1]
        output, h, c = self.step(x1, h, c)
        output = self.apply_mask(output, indices)
        probs = apply_temperature(output)
        x2 = torch.multinomial(probs, 1)
        max_vertex_id = 1
        prev_to_v = 1
        prev_verteces.add(1)

        dfs_codes.append([0, 1, 1, 1, 1])
        x = x2

        edge_dict = {}
        edge_dict[0] = set()
        edge_dict[0].add(1)
        edge_dict[1] = set()
        edge_dict[1].add(0)

        for i in range(max_len):
            end_token = self.tokenizer.END_TOKEN

            indices = []
            if prev_to_v == max_vertex_id:
                indices = (list(prev_verteces) + [max_vertex_id + 1] + [end_token])
            else:
                indices = (list(prev_verteces) + [end_token])


            indices.remove(prev_to_v)
            for v in edge_dict[prev_to_v]:
                indices.remove(v)

            output, h, c = self.step(x, h, c)
            output = self.apply_mask(output, indices)
            probs = apply_temperature(output)
            x1 = torch.multinomial(probs, 1)

#            print("x1:", x1.item())
            if x1.item() == end_token:
                break

            if x1.item() == prev_to_v + 1:
                dfs_codes.append([prev_to_v, prev_to_v + 1, 1, 1, 1])
                max_vertex_id = max(max_vertex_id, prev_to_v + 1)
                prev_verteces.add(prev_to_v)
                prev_verteces.add(prev_to_v + 1)

                if prev_to_v not in edge_dict:
                    edge_dict[prev_to_v] = set()
                edge_dict[prev_to_v].add(prev_to_v + 1)
                if prev_to_v + 1 not in edge_dict:
                    edge_dict[prev_to_v + 1] = set()
                edge_dict[prev_to_v + 1].add(prev_to_v)
                prev_to_v = prev_to_v + 1
                x = x1
            else:
                max_vertex_id = max(max_vertex_id, x1.item())
                new_prev_verteces = {v for v in prev_verteces if v < x1.item()}
                indices = (list(new_prev_verteces) + [max_vertex_id + 1])

                if x1.item() not in edge_dict:
                    print("error")
                    raise Exception("x1.item() not in edge_dict")

                for v in edge_dict[x1.item()]:
                    if v in indices:
                        indices.remove(v)

#                print("indices2: ", indices)
                output, h, c = self.step(x1, h, c)
                output = self.apply_mask(output, indices)
                probs = apply_temperature(output)
                x2 = torch.multinomial(probs, 1)

#                print("x2:", x2.item())

                dfs_codes.append([x1.item(), x2.item(), 1, 1, 1])

                if x1.item() not in edge_dict:
                    edge_dict[x1.item()] = set()
                if x2.item() not in edge_dict:
                    edge_dict[x2.item()] = set()
                edge_dict[x1.item()].add(x2.item())
                edge_dict[x2.item()].add(x1.item())

                prev_verteces.add(x1.item())
                prev_verteces.add(x2.item())
                max_vertex_id = max(max_vertex_id, x1.item())
                max_vertex_id = max(max_vertex_id, x2.item())
                prev_to_v = x2.item()

                x = x2
        return dfs_codes

    def load_model(self, model_path):
        """
        モデルを指定されたパスからロードします。

        model_path: モデルのパス
        """
        checkpoint = torch.load(model_path, map_location=get_device())
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデルが {model_path} から正常にロードされました。")

    def save_model(self, model_path):
        """
        モデルを指定されたパスに保存します。

        model_path: モデルのパス
        """
        torch.save({'model_state_dict': self.state_dict()}, model_path)
        print(f"モデルが {model_path} に正常に保存されました。")

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
            G = nx.Graph()
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
                avg_clustering = nx.average_clustering(G)
                graph_stats['clustering_coefficients'].append(avg_clustering)
            except:
                pass
            
            # 平均最短経路長
            try:
                avg_path = nx.average_shortest_path_length(G)
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
                assortativity = nx.degree_assortativity_coefficient(G)
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