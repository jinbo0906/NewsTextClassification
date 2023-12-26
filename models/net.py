from numbers import Number
from torch import nn
import torch
from transformers import BertModel, BertConfig
from .utils import model_decorator


@model_decorator
class BERTClass(torch.nn.Module):
    def __init__(self, conf, config_path, pretrained_path):
        super(BERTClass, self).__init__()
        self.param = conf["bert"]
        self.config = BertConfig.from_pretrained(config_path, output_hidden_states=True)
        self.l1 = BertModel.from_pretrained(pretrained_path, config=self.config)
        self.bilstm1 = torch.nn.LSTM(self.param["lstm_in_dim"], self.param["lstm_hid_dim"], self.param["lstm_layer_n"], bidirectional=True)
        self.l2 = torch.nn.Linear(128, 64)
        self.a1 = torch.nn.ReLU()
        self.l3 = torch.nn.Dropout(0.3)
        self.l4 = torch.nn.Linear(64, 14)

    def forward(self, ids, mask, token_type_ids):
        model_out = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        sequence_output, hidden_states = model_out.last_hidden_state, model_out.hidden_states
        # [bs, 200, 256]  [bs,256]
        bs = len(sequence_output)
        h12 = hidden_states[-1][:, 0].view(1, bs, 256)
        h11 = hidden_states[-2][:, 0].view(1, bs, 256)
        concat_hidden = torch.cat((h12, h11), 2)
        x, _ = self.bilstm1(concat_hidden)
        x = self.l2(x.view(bs, 128))
        x = self.a1(x)
        x = self.l3(x)
        output = self.l4(x)
        return output
