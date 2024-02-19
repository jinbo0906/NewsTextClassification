# NLP-新闻文本分类
天池 零基础入门NLP-新闻文本分类

熟悉Transformer，Bert


# TODO List

- [x] baseline


# Training Tasks

- [x] basic,num_workers=32：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Basic run_conf.train_conf.num_workers=32 hydra.job.chdir=True
- basic: num_workers=32,dropout=0,lstm_layer_n=1
- [ ] basic：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Basic hydra.job.chdir=True
- [ ] basic,lstm_layer_n=2：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Basic model_conf.conf.bert.lstm_layer_n=2 hydra.job.chdir=True


# Results

|         Model         | train_f1 | val_f1 | test_f1 |               model_path               |
|:---------------------:|:--------:|:------:|:-------:|:--------------------------------------:|
| basic,num_workers=32  |  0.9723  | 0.9248 | 0.9231  | 2023-12-27/09-46-29/Bert_lstm_5900.pth |
|         basic         |  0.985   | 0.9295 | 0.9249  |   2023-12-27/15-26-59/Bert_lstm_.pth   |
| basic,lstm_layer_n=2  |  0.9766  | 0.922  |         |   2023-12-27/13-29-19/Bert_lstm_.pth   |

# pretrain-mydataset

预训练Bert，首先考虑模型的输入输出格式，以及数据加载方式等。

**数据：**

首先对数据进行预处理，整合训练集和测试集语料，将比赛的文本转换成预训练需要的格式，即：

确保文本语料库是一行一行的句子，如下所示：

> Mr. Cassius crossed the highway, and stopped suddenly.
Something glittered in the nearest red pool before him.
Gold, surely!

**Tokenization：**

训练我们自己的tokenizer并为自己的语料库建立一个词汇表。从tokenizer库中选择BertWordPieceTokenizer，定义参数vocab_size=7000。模型以`vocab.txt`文件的形式保存到输出目录中。

**Model：**

接下来定义模型。首先，定义BERT模型的配置，其中Vocab_size是vocab大小，其余参数参考bert-base-cased模型参数。

```Python
config = BertConfig(
    vocab_size=6824,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    max_position_embeddings=512
)
```

然后创建一个BertForMaskedLM类的实例，专门用于掩码语言模型（MLM）任务。使用前面定义的config对象来初始化模型

```Python
model = BertForMaskedLM(config)
```

然后创建一个DataCollatorForLanguageModeling类的实例，语言建模任务的数据整理器。其中tokenizer=tokenizer将之前创建的tokenizer对象传递给数据整理器；mlm=True启用掩码语言模型（MLM）模式，这意味着数据整理器会在输入数据上执行掩码操作；mlm_probability=0.15设置在输入数据中进行掩码操作的概率为15%。

```Python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
```

最后基于transformers库中的TrainingArguments和Trainer类来配置模型训练的参数并实例化一个训练器。

参考链接：[https://www.kaggle.com/code/arnabs007/pretrain-a-bert-language-model-from-scratch/notebook](https://www.kaggle.com/code/arnabs007/pretrain-a-bert-language-model-from-scratch/notebook)