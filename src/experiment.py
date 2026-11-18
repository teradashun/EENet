import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .local import early_test


def full_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def shared_state(model, prefixes):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def evaluate_clients(global_model, client_states, test_loaders, device, evaluation_exits, prefixes):
    """各端末の個人層と最新の共有層を組み合わせ、指定Exitで評価する。"""
    current_shared = shared_state(
        global_model, prefixes
    )
    evaluation_model = copy.deepcopy(global_model)
    accuracies = []

    for i in range(len(client_states)):
        evaluation_model.load_state_dict(
            client_states[i],
            strict=True,
        )
        evaluation_model.load_state_dict(
            current_shared,
            strict=False,
        )

        accuracies.append(
            test_exit(
                evaluation_model,
                test_loaders[i],
                device,
                evaluation_exits[i],
            )
        )

    return accuracies