# FL：global_rounds vs accuracy
# Exit 1 ~ 4 全てからの平均精度を算出し、グラフ化する   

import copy
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import set_seed, choose_clients, select_optimizer, load_config, calculate_percentile_thresholds
from src.dataset import get_datasets, split_dataset, split_client_tests
from src.local import train, test
from src.server import federated_learning
from src.flops_counter import get_model_complexity_info
from src.models import eenet20_half, eenet34_1525_quarter, eenet34_51015_quarter


def build_global_model(model_name, params, device):
    if model_name == "eenet20_half":
        model = eenet20_half(num_classes=params["num_classes"]).to(device)
    elif model_name == "eenet34_1525_quarter":
        model = eenet34_1525_quarter(num_classes=params["num_classes"]).to(device)
    elif model_name == "eenet34_51015_quarter":
        model = eenet34_51015_quarter(num_classes=params["num_classes"]).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.train()
    return model


def evaluate_all_exits(model, dataloader, device):
    model.eval()
    corrects = []
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # exit_threshold=None を明示的に渡すことで全Exitの予測リストを取得
            outputs = model(inputs) 
            
            if not corrects:
                corrects = [0] * len(outputs)
            
            for i, out in enumerate(outputs):
                _, predicted = out.max(1)
                corrects[i] += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
    return [100.0 * c / total for c in corrects]


def get_saved_model_dir(save_root, model_name, dirichlet, cohort):
    """saved_models/{model_name}/{dirichlet}/{cohort} のパスを組み立てる"""
    return os.path.join(
        save_root,
        "saved_models",
        str(model_name),
        str(dirichlet),
        str(cohort)
    )


def save_model(model, model_name, dirichlet, cohort, params, save_root):
    """訓練済みモデルを saved_models/{model_name}/{dirichlet}/{cohort}/ に保存する。"""
    save_dir = get_saved_model_dir(save_root, model_name, dirichlet, cohort)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "params": params,
        "state_dict": model.state_dict(),
    }

    save_path = os.path.join(save_dir, "FedAvg.pth")
    torch.save(checkpoint, save_path)
    print(f"モデルを保存しました: {save_path}")
    return save_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # FedAvg/src
    parent_dir = os.path.dirname(current_dir)                  # FedAvg
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size    = config['training']['batch_size']
    epochs        = config['training']['epochs']
    lr            = config['training']['lr']
    optim_name    = config['training']['optim_name']
    model_name    = config['training']['model_name']
    dataset_name  = config['training']['dataset']
    num_ee        = config['training']['num_ee']
    loss_func     = config['training']['loss_func']
    temperature   = config['training']['temperature']

    global_rounds = config['FL']['global_rounds']
    num_clients   = config['FL']['num_clients']
    dirichlet     = config['FL']['dirichlet']
    ite_num       = config['FL']['ite_num']
    cohort        = config['FL']['cohort']
    mode          = config['FL']['mode']

    is_exit_model = ["eenet20_half", "eenet34_1525_quarter", "eenet34_51015_quarter"]
    if model_name in is_exit_model:
        is_exit = True
    else:
        is_exit = False
    
    # 最終Exit分を含めない
    intermediate_exit_nums = num_ee - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, _, val_loader, test_loader = get_datasets(batch_size, dataset_name, val_ratio=0.0)

    num_classes = 10 if dataset_name in ["MNIST", "CIFAR10"] else 100
    input_shape = (1, 28, 28) if dataset_name == "MNIST" else (3, 32, 32)

    params = {
        "num_ee": intermediate_exit_nums,
        "loss_func": loss_func,
        "num_classes": num_classes,
        "input_shape": input_shape,
    }

    acc_history = []

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")
        seed = ite + 100
        set_seed(seed)

        global_model = build_global_model(model_name, params, device)

        subsets = split_dataset(
            train_dataset, num_clients, dirichlet
        )

        test_subsets, train_counts, test_counts = split_client_tests(
            subsets, test_loader.dataset, seed=seed,
            num_classes=10, mode=mode,
        )

        client_loaders = [
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + i),
            )
            for i, subset in enumerate(subsets)
        ]
        
        test_loaders = [
            DataLoader(subset, batch_size=batch_size, shuffle=False)
            for subset in test_subsets
        ]

        set_seed(seed)
        chosen = [
            choose_clients(num_clients, cohort)
            for _ in range(global_rounds)
        ]
        ite_acc = []

        for round_idx in range(global_rounds):

            print(f"Round {round_idx+1}/ {global_rounds}")

            set_seed(seed + round_idx)
            participants = chosen[round_idx]
            client_updates = []
            for client_idx in participants:
                client_loaders[client_idx].generator.manual_seed(
                    seed + round_idx + client_idx
                )
                local_model = copy.deepcopy(global_model)
                local_optimizer = select_optimizer(
                    optim_name,
                    local_model,
                    lr
                    )

                # ローカルでモデルを訓練
                for _ in range(epochs):
                    train(local_optimizer, local_model, client_loaders[client_idx], device, model_name, loss_func, temperature)

                client_updates.append(local_model.state_dict())

            weights = [len(subsets[i]) for i in participants]
            federated_learning(client_updates, global_model, weights=weights)

            test_accs = evaluate_all_exits(
                global_model, test_loader, device
            )
            ite_acc.append(test_accs)

            formatted_accs = [f"{acc:.2f}" for acc in test_accs]
            print(f"Test Accuracies for all exits: {formatted_accs}")
        
        acc_history.append(ite_acc)
        
        if ite == 0:
            save_model(global_model, model_name, dirichlet, cohort, params, parent_dir)
    
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=0)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        f"model={model_name}",
        f"α={dirichlet}",
        f"K={cohort}"
        )
    os.makedirs(save_dir, exist_ok=True)
    
    # csv出力
    df_dict = {"round": np.arange(1, global_rounds + 1)}
    for i in range(num_ee):
        df_dict[f"Exit_{i+1}_accuracy"] = final_acc[:, i]
    df = pd.DataFrame(df_dict)
    csv_path = os.path.join(save_dir, "FedAvg.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))
    rounds = np.arange(1, global_rounds + 1) # グラフのX軸（ラウンド数）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i in range(num_ee):
        label_name = f'Exit {i+1} (Final)' if i == num_ee - 1 else f'Exit {i+1}'
        plt.plot(rounds, final_acc[:, i], linestyle='-', color=colors[i % len(colors)], label=label_name)

    # グラフのタイトルとラベル
    plt.title('FedAvg Accuracy by Exit', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, global_rounds)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, "FedAvg.png")
    plt.savefig(combined_path)
        
