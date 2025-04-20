import csv
import statistics
import sys

def calculate_statistics(filename):
    lengths = []
    count = 0

    try:
        # ファイルを開く（エンコーディングは必要に応じて変更してください）
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    second_column = row[1]
                    lengths.append(len(second_column))
                    count += 1
                else:
                    print(f"警告: 行が2列未満です: {row}")
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

    if count == 0:
        print("データがありません。")
        return

    average_length = statistics.mean(lengths)
    if count > 1:
        std_dev = statistics.stdev(lengths)
    else:
        std_dev = 0.0  # 標準偏差を計算できない場合

    print(f"2列目の文字列の平均長: {average_length:.2f}文字")
    print(f"2列目の文字列の標準偏差: {std_dev:.2f}文字")


if __name__ == "__main__":
    # 読み込むファイル名を指定してください
    filename = '/home/tabei/Dat/software/GxRNN/datasets/LINCS/mcf7.csv'  # 例: 'data.csv'

    calculate_statistics(filename)
    
