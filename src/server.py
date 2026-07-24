import copy
import torch

 
def average_models(client_updates, weights=None):
    """
    FedAvg の重み平均。
 
    重要: client_updates の中身を破壊しない（in-place 加算をしない）。
    これにより state_dict 内に共有テンソル（重複キー）があっても正しく動く。
    """
    K = len(client_updates)
    if weights is None:
        weights = [1.0 / K] * K
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    avg_state = {}
 
    for key in client_updates[0].keys():
        ref = client_updates[0][key]
        if not torch.is_floating_point(ref):
            avg_state[key] = ref.detach().clone()
            continue
 
        acc = ref.detach().clone().float() * weights[0]        # clone: 元を壊さない
        for i in range(1, K):
            acc = acc + client_updates[i][key].detach().float() * weights[i]   # 非 in-place
        avg_state[key] = acc.to(ref.dtype)
 
    return avg_state
 
 
def federated_learning(client_updates, global_model, weights=None):
    avg_state = average_models(client_updates, weights)
    global_model.load_state_dict(avg_state)
 
    return global_model
 

def report_shared_params(model, verbose=True):
    """state_dict 内で同じ実データを指しているキーのグループを返す。"""
    sd = model.state_dict()
    by_ptr = {}
    for k, v in sd.items():
        if v.numel() == 0:          # 空テンソルは data_ptr が 0 になるので除外
            continue
        by_ptr.setdefault(v.data_ptr(), []).append(k)
    groups = [ks for ks in by_ptr.values() if len(ks) > 1]

    if verbose:
        if not groups:
            print("共有テンソルなし（state_dict のキーはすべて独立）")
        else:
            print(f"★ 共有テンソルを {len(groups)} 組検出（重複キー）:")
            for ks in groups:
                print("   " + "  ==  ".join(ks))
            n_dup = sum(len(ks) - 1 for ks in groups)
            print(f"   重複キー総数: {n_dup}")
            print("   → 旧 average_models ではこれらが (2K-1)/K 倍に膨張していた。")
    return groups