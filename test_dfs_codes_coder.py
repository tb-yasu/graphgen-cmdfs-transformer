
from util import Tokenizer
from dataloader import DFSDataset


def test_dfs_codes(filename_dfs_codes: str, filename_csv: str):

    tokenizer   = Tokenizer()
    dfs_dataset = DFSDataset(tokenizer, filename_csv, filename_dfs_codes)

#    for i in range(len(dfs_dataset)):
#        features, dfs_codes = dfs_dataset[i]
#        print(features)
#        print(dfs_codes)
#        break

if __name__ == "__main__":
#    filename_dfs_codes = "/home/tabei/Dat/software/GxRNN/datasets/LINCS/mcf7-dfscodes.txt"
    filename_dfs_codes = "/home/tabei/prog/cpp/gspan/output.txt"
    filename_csv = "/home/tabei/Dat/software/GxRNN/datasets/LINCS/mcf7.csv"
    test_dfs_codes(filename_dfs_codes, filename_csv)