import argparse
from dataloader import DFSDataLoader
from util import Tokenizer
from util import *
from vec2graph_trans import TransformerGraphGenerator
from torch import nn
from util import check_minimality
from util import is_valid_dfs_code
from util import calculate_novelty
from util import calculate_uniqueness
from util import calculate_diversity
from util import GraphMetrics
from util import evaluate_generated_graphs
import time
import networkx as nx
from stats import orbit_stats_all, motif_stats, clustering_stats, degree_stats
from util import compute_frechet_distance_for_graph_sets

def train_model(train_loader, test_loader, tokenizer, args):
    torch.cuda.reset_peak_memory_stats()

#    vec2seq = Vec2SeqPeeky(tokenizer, args.emb_size, args.hidden_size, args.latent_vec_size, args.num_layers, args.dropout).to(get_device())
    vec2seq = TransformerGraphGenerator(tokenizer, args.emb_size, args.hidden_size, args.num_layers, args.num_heads, args.dropout, args.max_len).to(get_device())

    nll_loss = nn.NLLLoss()

    optimizer = torch.optim.Adam(
        params=vec2seq.parameters(), 
        lr=args.learning_rate
    )

    with open(args.train_results, 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{},{}n'.format(\
            'Epoch', 
            'Loss', 
            'Total', 
            'Valid', 
            'Valid_rate'
        ))

    total_learning_time = 0
    total_test_time = 0
    total_validation_time = 0

    train_loss = []

    print("get_device(): ", get_device())
    
    for epoch in range(args.epochs):
        vec2seq.train()
        total_loss = 0

        start_time = time.time()
        for batch in train_loader:
            dfs_codes = batch[0]
            dfs_sequences = batch[1]

            dfs_sequences = torch.stack([torch.tensor(d, dtype=torch.long) for d in dfs_sequences]).to(get_device())
            decoder_inputs = dfs_sequences[:, :-1]
            output = vec2seq(decoder_inputs)

            # ラベル値が範囲外の場合エラーを発生させる
#            assert dfs_sequences.min() >= 0, "Error: Negative labels found!"
#            assert dfs_sequences.max() < num_classes, f"Error: Label value {dfs_sequences.max().item()} exceeds num_classes ({num_classes})."


            loss = nll_loss(output, dfs_sequences[:, 1:].contiguous().view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss.append(total_loss)
        end_time = time.time()
        total_learning_time += end_time - start_time


        print("Epoch: {}, Training Loss: {:.2f}".format(epoch, total_loss))
        print("Total learning time: {:.2f} seconds".format(total_learning_time))
        if torch.cuda.is_available():
            # ピークメモリの使用量
            print("Peak memory allocated:", torch.cuda.max_memory_allocated())
            # GPU上で使用中のメモリの総量 (バイト単位)
            print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
            # GPU上でキャッシュに保持されているメモリの総量
            print(f"Memory cached: {torch.cuda.memory_reserved()} bytes")
            # メモリの要約を表示
#            print(torch.cuda.memory_summary())
        else:
            print("Running on CPU - memory statistics not available")

#        if epoch % 100 != 0:
#            continue
        
    start_time = time.time()

#    num_samples = min(args.num_samples, len(test_loader.dataset))
    num_samples = len(test_loader.dataset)
    print("num_samples: ", num_samples)

    if args.disable_label == False:
        print("sample")
        generated_graphs = vec2seq.sample(num_samples, args.max_len, args.temperature)
    else:
        print("sample_no_label")
        generated_graphs = vec2seq.sample_no_label(num_samples, args.max_len, args.temperature)

    end_time = time.time()
    total_test_time += end_time - start_time
    print("Total test time: {:.2f} seconds".format(total_test_time))    
        
    start_time = time.time()

    # DFSコードからnx.Graphオブジェクトに変換
    nx_graphs = []
    for dfs_codes in generated_graphs:
        G = nx.Graph()
        for code in dfs_codes:
            from_v, to_v, from_label, edge_label, to_label = code
            if from_v not in G.nodes():
                G.add_node(from_v, label=from_label)
#                G.add_node(from_v)
            if to_v not in G.nodes():
                G.add_node(to_v, label=to_label)
#                G.add_node(to_v)
            G.add_edge(from_v, to_v, label=edge_label)
#            G.add_edge(from_v, to_v)
        nx_graphs.append(G)

    # test_loaderからリファレンスグラフを取得してnx.Graphに変換
    reference_graphs = []
    for batch in test_loader:
        dfs_codes_batch = batch[0]  # バッチ内のDFSコードを取得
        for dfs_codes in dfs_codes_batch:
            G = nx.Graph()
            for code in dfs_codes:
                from_v, to_v, from_label, edge_label, to_label = code
                if from_v not in G.nodes():
                    G.add_node(from_v, label=from_label)
                    #G.add_node(from_v)
                if to_v not in G.nodes():
                    G.add_node(to_v, label=to_label)
#                    G.add_node(to_v)
                G.add_edge(from_v, to_v, label=edge_label)
#                G.add_edge(from_v, to_v)
            reference_graphs.append(G)

    if args.disable_label == True:
        for G in nx_graphs:
            for node in G.nodes():
                G.nodes[node]['label'] = 1
            for edge in G.edges():
                G.edges[edge]['label'] = 1
        for G in reference_graphs:
            for node in G.nodes():
                G.nodes[node]['label'] = 1
            for edge in G.edges():
                G.edges[edge]['label'] = 1

    print("reference_graphs: ", len(reference_graphs))
    print("nx_graphs: ", len(nx_graphs))

    degree_dist = degree_stats(reference_graphs, nx_graphs)
    clustering_dist = clustering_stats(reference_graphs, nx_graphs)
    orbit_dist = orbit_stats_all(reference_graphs, nx_graphs)

    print("==== MMD Results ====")
    print(f"Degree MMD:      {degree_dist:.4f}")
    print(f"Clustering MMD:  {clustering_dist:.4f}")
    print(f"Orbit MMD(4-node motif, 11 classes): {orbit_dist:.4f}")

    compare = Compare(tokenizer, args.wl_iteration)
    mmd = compare.comp_mmd(reference_graphs, nx_graphs)
#        mmd_subsample = compare.compute_mmd_multi_dim_subsample(test_loader.dataset, generated_graphs)
    print(f"Graph KernelMMD: {mmd:.4f}")

    fd_value = compute_frechet_distance_for_graph_sets(reference_graphs, nx_graphs)
    print("Fréchet Distance between set1 and set2: ", fd_value)

    end_time = time.time()
    total_validation_time += end_time - start_time

    print("Total validation time: {:.2f} seconds".format(total_validation_time))

def main():
    parser = argparse.ArgumentParser(description="Train a vec2graph model")
    parser.add_argument('--input_file_dfs', type=str, help="input file name")
    parser.add_argument('--emb_size', type=int, default=128, help="embedding size")
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden size")
    parser.add_argument('--num_layers', type=int, default=3, help="number of layers")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout probability")
    parser.add_argument('--num_heads', type=int, default=8, help="number of heads")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--train_results', type=str, default='train_results.csv', help="training results file")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--train_rate', type=float, default=0.9, help="train rate")
    parser.add_argument('--max_len', type=int, default=100, help="max length of dfs sequence")
    parser.add_argument('--wl-iteration', type=int, default=5, help="iteration of weisfeiler-lehman")
    parser.add_argument('--num_samples', type=int, default=1000, help="number of samples")
    parser.add_argument('--temperature', type=float, default=2.0, help="temperature")
    parser.add_argument('--disable_label', action='store_true', help="Disable label usage")

    args = parser.parse_args()
    print(args)


    tokenizer = Tokenizer()
    dfs_loader = DFSDataLoader(tokenizer, args.input_file_dfs, args.batch_size, args.train_rate, args.disable_label)
    train_loader, test_loader = dfs_loader.get_dataloader()

    trained_model = train_model(train_loader, test_loader, tokenizer, args)
    

if __name__ == "__main__":
    main()