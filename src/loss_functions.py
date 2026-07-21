import os
import sys
import torch
import numpy as np
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))   # EE_FL/src
parent_dir = os.path.dirname(current_dir)                  # EE_FL
sys.path.append(parent_dir)

def loss(loss_func, device, pred, target):
    """loss function

    Arguments are
    * loss_func:    the loss function to use.
    * pred:         prediction result of each exit point.
    * target:       target prediction values.

    This function switches between the loss functions.
    """
    # Exitモデルではない場合の処理
    if torch.is_tensor((pred)):
        pred_loss = F.cross_entropy(pred, target)
        return pred_loss, pred_loss, torch.tensor(0.0, device=device)

    if loss_func == 'v1':
        return loss_v1(loss_func, pred, target)

    if loss_func == 'v2':
        return loss_v2(loss_func, pred, target)


def loss_v1(loss_func, pred, target):
    criterion = torch.nn.CrossEntropyLoss()
    pred_loss = 0

    for i in range(len(pred)):
        loss = criterion(pred[i], target)
        pred_loss += loss

    pred_loss /= len(pred)

    return pred_loss


def loss_v2(loss_func, pred, target):
    criterion = torch.nn.CrossEntropyLoss()

    base_arr = np.arange(1, len(pred) + 1)
    arr = base_arr / np.sum(base_arr)

    pred_loss = 0

    for i, weight in enumerate(arr):
        loss = criterion(pred[i], target)
        pred_loss += loss * weight

    return pred_loss