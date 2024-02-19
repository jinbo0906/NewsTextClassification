import tokenizers
# from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, LineByLineTextDataset


def train_tokenizer(file_path, save_path):
    bwpt = tokenizers.BertWordPieceTokenizer()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bwpt.add_special_tokens(special_tokens)
    bwpt.train(
        files=file_path,
        vocab_size=7000,
        min_frequency=2
    )
    bwpt.save_model(save_path)


if __name__ == "__main__":

    # # file_path = "../dataset/train_my_0.txt"
    # # save_path = "bert-mini-my"
    # # train_tokenizer(file_path, save_path)
    vocab_file_dir = 'bert-mini-my/vocab.txt'
    # vocab_file_dir = 'bert-base-cased/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

    sentence = '2967 6758 339 2021 1854 3731 4109 3792 4149 1519 2058 3912 2465 2410 1219 6654 7539 264'
    encoded_input = tokenizer.tokenize(sentence)
    print(encoded_input)

