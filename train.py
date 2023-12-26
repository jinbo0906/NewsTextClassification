import numpy as np
import time
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import hydra
from hydra.utils import get_original_cwd
import queue
import logging
import random
import pandas as pd
from omegaconf import OmegaConf
from sklearn import metrics

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from data import MyDataset
from models import get_model, model_saver


def global_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class TrainingSystem:

    def __init__(self, conf):

        self.log = logging.getLogger("Train")

        self.global_conf = conf["global_conf"]
        self.model_conf = conf["model_conf"]
        self.run_conf = conf["run_conf"]
        self.data_conf = conf["data_conf"]

        self.tensorboard_writer = SummaryWriter("./tensorboard_log")
        self.log.info(OmegaConf.to_yaml(conf))

        # --------------
        # global
        # --------------
        self.log.info("init global conf...")

        global_seed(self.global_conf["seed"])  # seed
        self.device = torch.device(self.global_conf["device"])  # device
        self.log.info(f"Device: {self.device}")
        self.project_root = get_original_cwd()  # root

        # --------------
        # data
        # --------------
        self._data_init()

        # --------------
        # model
        # --------------
        self._model_init()

        # -------------
        # optimizer
        # -------------
        self._optim_init()

        # ------------
        # loss func
        # ------------
        self._loss_init()

        # ------------
        # others
        # -------------
        self.min_val_score = float('inf')
        self.best_model_path = None
        self.model_save_queue = queue.Queue(maxsize=5)

    def _data_init(self):
        self.log.info("init data...")
        # 观测数据
        observe_data_conf = self.data_conf["observe_data"]
        train_data_path = os.path.join(self.project_root, "dataset", "train_set.csv")

        # test data
        test_data_path = os.path.join(self.project_root, "dataset", "test_a.csv")

        # eval data
        n_eval_sections = int(self.data_conf["observe_data"]["n_eval_sections"])
        train_size = self.data_conf["observe_data"]["train_size"]

        if n_eval_sections == 0:
            val_type = "ratio"
        else:
            val_type = "section"

        if val_type == "section":
            observe_data = pd.read_csv(train_data_path, sep='\t')
            test_data = pd.read_csv(test_data_path, sep='\t')
            test_data['label'] = 0
            # random select
            train_data = observe_data.sample(frac=train_size, random_state=7)
            val_data = observe_data.drop(train_data.index).reset_index(drop=True)
            train_data = train_data.reset_index(drop=True)
        else:
            observe_data = pd.read_csv(train_data_path, sep='\t')
            test_data = pd.read_csv(test_data_path, sep='\t')
            test_data['label'] = 0
            # random select
            train_data = observe_data.sample(frac=train_size, random_state=7)
            val_data = observe_data.drop(train_data.index).reset_index(drop=True)
            train_data = train_data.reset_index(drop=True)
        self.log.info("full Dataset: {}".format(observe_data.shape))
        self.log.info("train Dataset: {}".format(train_data.shape))
        self.log.info("valid Dataset: {}".format(val_data.shape))
        self.log.info("test Dataset: {}".format(train_data.shape))

        tokenizer_path = os.path.join(self.project_root, "pretrained", "bert-mini")
        # tokenizer_path = "E:/code/NewsTextClassification/pretrained/bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        train_set = MyDataset(train_data, tokenizer, self.data_conf["observe_data"]["max_len"])
        valid_set = MyDataset(val_data, tokenizer, self.data_conf["observe_data"]["max_len"])
        test_set = MyDataset(test_data, tokenizer, self.data_conf["observe_data"]["max_len"])
        train_params = {'batch_size': self.run_conf["train_conf"]["batch_size"], 'shuffle': True, 'num_workers': self.run_conf["train_conf"]["num_workers"], 'pin_memory': True}
        valid_params = {'batch_size': self.run_conf["train_conf"]["batch_size"], 'shuffle': True, 'num_workers': self.run_conf["train_conf"]["num_workers"], 'pin_memory': True}
        test_params = {'batch_size': self.run_conf["train_conf"]["batch_size"], 'shuffle': False, 'num_workers': self.run_conf["train_conf"]["num_workers"], 'pin_memory': True}

        self.train_loader = DataLoader(train_set, **train_params)
        self.valid_loader = DataLoader(valid_set, **valid_params)
        self.test_loader = DataLoader(test_set, **test_params)

    def _model_init(self):
        self.log.info("init model...")
        config_path = os.path.join(self.project_root, "pretrained", "bert-mini", "config.json")
        pretrained_path = os.path.join(self.project_root, "pretrained", "bert-mini", "pytorch_model.bin")
        self.model = get_model(self.model_conf, config_path, pretrained_path)
        self.model = self.model.to(self.device)
        self.log.info(self.model)
        if self.model_conf["load_checkpoint"]:
            self.model.load_state_dict(torch.load(os.path.join(self.project_root, self.model_conf["checkpoint_path"])))

    def _optim_init(self):
        self.log.info("init optimizer...")
        if self.run_conf["optim"] == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), **self.run_conf["optim_conf"])
        else:
            raise ValueError("optim: {} is not supported".format(self.run_conf["optim"]))

        if self.run_conf["sch"] == "Identify":
            self.sch = None
        elif self.run_conf["sch"] == "step":
            step_sch_conf = self.run_conf["sch_step"]
            self.sch = torch.optim.lr_scheduler.StepLR(self.optim, step_size=int(
                self.run_conf["train_conf"]["epoch"] / step_sch_conf["stage"]),
                                                       gamma=step_sch_conf["gamma"])
        else:
            raise ValueError("sch: {} is not supported".format(self.run_conf["sch"]))

    def _loss_init(self):
        self.log.info("init loss...")
        self.observe_loss = torch.nn.CrossEntropyLoss()  # 选择损失函数

    def train_loop(self):
        self.log.info("begin training...")
        num_epochs = self.run_conf["train_conf"]["epoch"]
        val_score = 0.0
        step_num = 0
        for epoch in range(num_epochs):
            self.log.info("Epoch_{} begin".format(epoch))
            train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            loop_bar = tqdm(self.train_loader)
            for data in loop_bar:
                if step_num % self.run_conf["train_conf"]["eval_freq"] == 0:
                    valid_acc, val_score = self.eval_loop(step_num)
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                targets = data['targets'].to(self.device)
                self.optim.zero_grad()
                y_hat = self.model(ids, mask, token_type_ids)
                loss = self.observe_loss(y_hat, targets.long())
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == targets))).float()
                loss.backward()
                self.optim.step()

                train_acc = train_acc_sum / len(self.train_loader)
                train_f1 = metrics.f1_score(targets.cpu().numpy().tolist(), torch.argmax(y_hat, dim=1).cpu().numpy().tolist(), average='macro')
                log_str = "train loss: {:.4e}  train acc: {:.4e}  train f1: {:.4e} val acc: {:.4e} val f1: {:.4e}".format(loss.item(), train_acc.item(), train_f1, valid_acc, val_score)
                loop_bar.set_description(log_str)
                loop_bar.refresh()
                if step_num % self.run_conf["train_conf"]["print_frequency"]:
                    self.log.info(f"step: {step_num}")
                    self.log.info(f"train loss: {loss.item()}")
                    self.log.info(f"train acc: {train_acc.item()}")
                    self.log.info(f"train f1: {train_f1}")
                # tensorboard
                self.tensorboard_writer.add_scalar("Loss/train_Loss", loss.item(), step_num)
                self.tensorboard_writer.add_scalar("Loss/train_acc", train_acc.item(), step_num)
                self.tensorboard_writer.add_scalar("Loss/train_f1", train_f1, step_num)
                step_num += 1
            self.sch.step()

        self.eval_loop(num_epochs)
        self.log.info("train down...")
        self.tensorboard_writer.close()

    def eval_loop(self, step):
        self.log.info("begin eval...")
        self.model.eval()

        is_save_model = False
        acc_sum, n = torch.tensor([0], dtype=torch.float32, device=self.device), 0
        y_pred_, y_true_ = [], []
        loop_bar = tqdm(self.valid_loader)

        with torch.no_grad():
            for batch_data in loop_bar:
                ids = batch_data['ids'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                token_type_ids = batch_data['token_type_ids'].to(self.device)
                targets = batch_data['targets'].to(self.device)
                n += targets.shape[0]
                y_hat = self.model(ids, mask, token_type_ids).to(self.device)

                y_pred_.extend(torch.argmax(y_hat, dim=1).cpu().numpy().tolist())
                y_true_.extend(targets.cpu().numpy().tolist())
        valid_f1 = metrics.f1_score(y_true_, y_pred_, average='macro')
        valid_acc = acc_sum.item()/ n
        self.tensorboard_writer.add_scalar("Rmsd loss/val acc", valid_acc, step)
        self.tensorboard_writer.add_scalar("Dis loss/valid f1", valid_f1, step)
        self.log.info(f"val step: {step}")
        self.log.info(f"val acc: {valid_acc}")
        self.log.info(f"val f1: {valid_f1}")

        if valid_f1 < self.min_val_score and step != 0:
            self.min_val_score = valid_f1
            is_save_model = True

        if not self.model_save_queue.full():
            is_save_model = True

        if is_save_model:
            self.best_model_path = model_saver(
                save_folder="./",
                model=self.model,
                save_name="NS",
                step=step
            )

            if self.model_save_queue.full():
                del_model_path = self.model_save_queue.get()
                os.remove(del_model_path)

            self.model_save_queue.put(self.best_model_path)

        self.model.train()

        return valid_acc, valid_f1

    def test_loop(self):
        # 预测模型
        self.log.info("begin predict...")
        self.log.info("load best model...")

        model_path = None
        while not self.model_save_queue.empty():
            model_path = self.model_save_queue.get()

        if model_path is not None:
            self.log.info("Load Best Model...")
            self.log.info("Best Model: {}".format(model_path))
            self.model.load_state_dict(torch.load(model_path))

        preds_list = []

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)
                batch_preds = list(self.model(ids, mask, token_type_ids).argmax(dim=1).cpu().numpy())
                for preds in batch_preds:
                    preds_list.append(preds)
        save_test_data_path = os.path.join(self.project_root, "dataset", "test_a_sample_submit.csv")
        submission = pd.read_csv(save_test_data_path)
        submission['label'] = preds_list
        submission.to_csv('submission.csv', index=False)
        self.log.info("test done...")


@hydra.main(version_base=None, config_path="conf", config_name="Basic")
def train_setup(cfg):
    train_system = TrainingSystem(cfg)

    if cfg["run_conf"]["main_conf"]["run_mode"] == "train":
        train_system.train_loop()
        train_system.test_loop()
    elif cfg["run_conf"]["main_conf"]["run_mode"] == "test":
        train_system.test_loop()
    else:
        raise ValueError("run_mode: {} is not supported".format(cfg["run_conf"]["main_conf"]["run_mode"]))


if __name__ == "__main__":
    train_setup()
