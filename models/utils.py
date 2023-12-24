import os
import torch
from torch import nn

model_dict = dict()


def model_decorator(model):
    model_dict[model.__name__] = model
    return model


def get_model(conf, config_path, pretrained_path):
    model_name = conf["name"]
    model_conf = conf["conf"]

    return model_dict[model_name](model_conf, config_path, pretrained_path)


def model_saver(save_folder, model, save_name, step=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if step is not None:
        save_path = os.path.join(save_folder, "{}_{}.pth".format(save_name, step))
    else:
        save_path = os.path.join(save_folder, "{}.pth".format(save_name))
    torch.save(model.state_dict(), save_path)
    return save_path
