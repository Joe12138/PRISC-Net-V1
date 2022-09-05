import os
import re
import csv
import logging
import shutil
import numpy as np
import torch
from typing import List


_INVALID_TAG_CHARACTERS = re.compile(r'[^-/\w\.]')


def to_numpy(variable, USE_CUDA):
    return variable.cpu().data.numpy() if USE_CUDA else variable.data.numpy()


def save_checkpoint(state, is_best, dirpath, filename='model_checkpoint.pth'):
    if(not os.path.exists(dirpath)):
        os.makedirs(dirpath)
    path = os.path.join(dirpath, filename)
    patht = path + '.tmp'
    if torch.__version__ >= '1.6.0':
        torch.save(state, patht, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, patht)
    shutil.move(patht, path)
    if is_best:
        shutil.copyfile(path, os.path.join(dirpath, 'model_best.pth'))


def load_checkpoint(dirpath, best=False):
    # Load the best or the lastest model from directory
    filename = 'model_best.pth' if best else 'model_checkpoint.pth'
    path = os.path.join(dirpath, filename)
    if os.path.exists(path):
        logging.info(f"Previous CHECKPOINT loaded from {path}")
        return torch.load(path)
    else:
        return None


def to_leagal_tf_summary_name(name):
    """
    This function replaces all illegal characters with _s, and logs a warning
    """
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        # if new_name != name:
        #     logging.info(f'Summary name {name} is illegal; using {new_name} instead.')
        name = new_name
    return name


def adjust_learning_rate(optimizer, epoch: int, decayed_epoch: List[int], init_lr: float, lr_decay_factor: float):
    """Decay the learning rate by the given factor at the given epoch list"""
    if epoch in decayed_epoch:
        decayed_epoch = sorted(decayed_epoch)  # Arrange epochs in descending order
        lr = init_lr * (lr_decay_factor ** (decayed_epoch.index(epoch) + 1))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
            if i == 0:
                logging.info(f"Learning Rate decays to {param_group['lr']}")