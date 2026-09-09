import json
import os
import traceback

from pytorch_lightning import Callback
import torch

def wlog(msg, file):
    if file is not None:
        with open(file, 'a') as f:
            print(msg, file=f)
    print(msg)

class NaNDataDetector(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            wlog(f"NaN/Inf loss at batch {batch_idx}",'nan_callback.log')
            torch.save(batch, f"nanckpt/error_batch_{batch_idx}.pt")
            trainer.should_stop = True  # 停止训练


def m_calc(p):
    """
    :param p: (px,py,pz,e)
    :return: inv. m
    """
    return (p[3] ** 2 - p[0] ** 2 - p[1] ** 2 - p[2] ** 2).sqrt().item()

def deep_update(base, update):
    """递归更新嵌套字典"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # 如果两个值都是字典，递归合并
            deep_update(base[key], value)
        else:
            # 否则直接替换/添加
            base[key] = value
    return base