import torch


if __name__ == "__main__":
    # 建立词典
    sentence = 'Life is short, eat dessert first'

    dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}

    print(dc)
    # 为每个词分配一个整数索引
    sentence_int = torch.tensor(
        [dc[s] for s in sentence.replace(',', '').split()]
    )
    print(sentence_int)
