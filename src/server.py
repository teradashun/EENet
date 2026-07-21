import copy
import torch
from src.norm_tools import reproject_model_

def average_models(client_updates):
    avg_state = copy.deepcopy(client_updates[0])
    for key in avg_state.keys():
        for i in range(1, len(client_updates)):
            avg_state[key] += client_updates[i][key]
        avg_state[key] = torch.div(avg_state[key], len(client_updates))
    return avg_state

"""
def federated_learning(client_updates, global_model):
    avg_state = average_models(client_updates)
    global_model.load_state_dict(avg_state)
    
    return global_model
"""


def federated_learning(client_updates, global_model, si_param_names=None, ref_norms=None):
    """
    FedAvg で集約 → global_model に反映。
    si_param_names / ref_norms を渡すと、集約直後にスケール不変 conv の
    ノルムだけを基準値へ再射影する（スケール不変なので損失は不変、暴走だけ消える）。
    両方 None なら従来どおりの素の FedAvg。
    """
    avg_state = average_models(client_updates)
    global_model.load_state_dict(avg_state)
 
    if si_param_names is not None and ref_norms is not None:
        reproject_model_(global_model, si_param_names, ref_norms)
 
    return global_model
