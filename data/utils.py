import numpy as np
import pandas as pd


def split_data(observe_data, val_ratio):
    """
    使用 pandas 读取的数据集进行分割
    sample() 函数的返回值是一个新的 DataFrame 对象，包含了打乱后的数据。

    Args:
        observe_data: 读取的数据集
        val_ratio: 验证集比例

    Returns:
        训练集、验证集
    """

    # 随机打乱数据
    observe_data = observe_data.sample(frac=1, random_state=42)

    # 计算验证集大小
    n_val = int(len(observe_data) * val_ratio)

    # 分割数据集
    val_data = observe_data[:n_val]
    train_data = observe_data[n_val:]

    return train_data, val_data


if __name__ == "__main__":
    # 读取数据集
    observe_data = pd.read_csv("../dataset/train_set.csv", sep="\t")

    # 分割数据集
    val_data, train_data = split_data(observe_data, 0.1)

    # 打印数据集大小
    print("验证集大小：", len(val_data))
    print("训练集大小：", len(train_data))
