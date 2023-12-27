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

|         Model         | train_f1 | val_f1  | test_f1  |               model_path               |
|:---------------------:|:--------:|:-------:|:--------:|:--------------------------------------:|
| basic,num_workers=32  |  0.9723  | 0.9248  |  0.9231  | 2023-12-27/09-46-29/Bert_lstm_5900.pth |
|         basic         |          |         |          |   2023-12-27/14-47-52/Bert_lstm_.pth   |
| basic,lstm_layer_n=2  |          |         |          |   2023-12-27/13-16-39/Bert_lstm_.pth   |
