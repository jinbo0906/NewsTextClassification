import os

import pandas as pd
import numpy as np


def create_raw_documents(n):
    # 读取 train_set 和 test_a 的 text 部分，每行中间插入空行
    files_in = ["../dataset/train_set.csv", "../dataset/test_a.csv"]
    files_out = []
    for i in range(n):
        files_out.append(open("../dataset/train_my_{}.txt".format(i), 'w'))

    for file in files_in:
        data = pd.read_csv(file, sep='\t')
        data = data['text'].tolist()
        data_num = len(data)
        max_len = data_num // n
        index = list(range(0, data_num, max_len))
        index.append(data_num)
        for i in range(len(index) - 1):
            files_out[i].writelines("\n\n".join(data[index[i]:index[i + 1]]))
            files_out[i].writelines("\n")
            files_out[i].writelines("\n")
    for f in files_out:
        f.close()


def create_vocab():
    files = ["../dataset/train_set.csv", "../dataset/test_a.csv"]
    # 定义文件路径
    file_path = 'bert-mini-my/vocab.txt'
    # 分离出目录部分
    directory = os.path.dirname(file_path)
    # 判断目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    my_open = open(file_path, 'w')
    word = set()
    for file in files:
        data = pd.read_csv(file, sep='\t')
        for text in data['text']:
            word.update(text.split())
    extra_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    my_open.writelines("\n".join(extra_tokens))
    my_open.writelines("\n")
    my_open.writelines("\n".join(word))
    my_open.close()


if __name__ == '__main__':
    create_raw_documents(1)
    # create_vocab()



