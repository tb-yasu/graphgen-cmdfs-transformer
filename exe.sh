#!/bin/bash

# Citeseer ego experiments
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.8 --learning_rate 0.001 --max_len 1100 > res_citeseer_ego_label_temperature_0.8_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.85 --learning_rate 0.001 --max_len 1100 > res_citeseer_ego_label_temperature_0.85_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.9 --learning_rate 0.001 --max_len=1100 > res_citeseer_ego_label_temperature_0.9_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.95 --learning_rate 0.001 --max_len 1100 > res_citeseer_ego_label_temperature_0.95_new.txt
#python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.98 --learning_rate 0.001 --max_len 1100 > res_citeseer_ego_label_temperature_0.98_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1 --learning_rate 0.001 --max_len 1100 > res_citeseer_ego_label_temperature_1_new.txt
#python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1.5 --learning_rate 0.001 --max_len=1100 > res_citeseer_ego_label_temperature_1.5_new.txt
#python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/citeseer_ego_labeled.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 2 --learning_rate 0.001 --max_len=1100 > res_citeseer_ego_label_temperature_2_new.txt

# Citeseer community experiments
python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.8 --learning_rate 0.001 --max_len 3000 > res_citeseer_community_label_temperature_0.8_new.txt
python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.85 --learning_rate 0.001 --max_len 3000 > res_citeseer_community_label_temperature_0.85_new.txt
python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.9 --learning_rate 0.001 --max_len 3000 > res_citeseer_community_label_temperature_0.9_new.txt
python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.95 --learning_rate 0.001 --max_len 3000 > res_citeseer_community_label_temperature_0.95_new.txt
python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1 --learning_rate 0.001 --max_len 3000 > res_citeseer_community_label_temperature_1_new.txt
#python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1.5 --learning_rate 0.001 --max_len=3000 > res_citeseer_community_label_temperature_1.5_new.txt
#python3 train.py --input_file_dfs  /home/tabei/Dat/graph_dataset/original/community_label.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 2 --learning_rate 0.001 --max_len=3000 > res_citeseer_community_label_temperature_2_new.txt

python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/enzymes.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.8 --learning_rate 0.001 --max_len 200 > res_enzymes_label_temperature_0.8_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/enzymes.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.85 --learning_rate 0.001 --max_len 200 > res_enzymes_label_temperature_0.85_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/enzymes.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.9 --learning_rate 0.001 --max_len 200 > res_enzymes_label_temperature_0.9_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/enzymes.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.95 --learning_rate 0.001 --max_len 200 > res_enzymes_label_temperature_0.95_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/enzymes.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1 --learning_rate 0.001 --max_len 200 > res_enzymes_label_temperature_1_new.txt

python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/protein.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.8 --learning_rate 0.001 --max_len 220 --num_samples 150 > res_proteins_label_temperature_0.8_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/protein.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.85 --learning_rate 0.001 --max_len 220 --num_samples 150 > res_proteins_label_temperature_0.85_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/protein.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.9 --learning_rate 0.001 --max_len 220 --num_samples 150 > res_proteins_label_temperature_0.9_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/protein.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 0.95 --learning_rate 0.001 --max_len 220 --num_samples 150 > res_proteins_label_temperature_0.95_new.txt
python3 train.py --input_file_dfs /home/tabei/Dat/graph_dataset/original/protein.dfs_codes --epochs 1001 --train_rate 0.8 --temperature 1 --learning_rate 0.001 --max_len 220 --num_samples 150 > res_proteins_label_temperature_1_new.txt







