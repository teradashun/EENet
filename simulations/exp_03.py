# エポック数 vs 精度(%)

import time
import copy
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from src.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet20, resnet32, resnet44, resnet56, resnet110
from src.eenet import eenet18, eenet34, eenet50, eenet101, eenet152, eenet20, eenet32, eenet44, eenet56, eenet110
from src.utils import set_seed, load_config, acc_average, select_optimizer, calculate_percentile_thresholds
from src.local import train, test, early_test
from src.server import federated_learning
from src.dataset import get_datasets, split_dataset


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # EE_FL/simulations
    parent_dir = os.path.dirname(current_dir)                  # EE_FL
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size          = config['training']['batch_size']
    epochs              = config['training']['epochs']
    lr                  = config['training']['lr']
    optim_name          = config['training']['optim_name']
    model_name          = config['training']['model_name']
    dataset             = config['training']['dataset']
    num_ee              = config['training']['num_ee']
    distribution        = config['training']['distribution']
    loss_func           = config['training']['loss_func']
    zero_init_residual  = config['training']['zero_init_residual']
    exit_plot_num       = config['training']['exit_plot_num']

    ite_num             = config['FL']['ite_num']

    is_exit_model = ["eenet18", "eenet34", "eenet50", "eenet101", "eenet152", "eenet20", "eenet32", "eenet44", "eenet56", "eenet110"]
    if model_name in is_exit_model:
        is_exit = True
    else:
        is_exit = False
    
    # 最終Exit分を含めない
    intermediate_exit_nums = num_ee - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, train_loader, val_loader, _ = get_datasets(batch_size, dataset, val_ratio=0.2)
    _, _, _, speed_test_loader = get_datasets(1, dataset, val_ratio=0.2)

    num_classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100

    input_shape = (1, 28, 28) if dataset == "MNIST" else (3, 32, 32)

    params = {
        "num_ee": intermediate_exit_nums,
        "distribution": distribution,
        "loss_func": loss_func,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "zero_init_residual": zero_init_residual,
    }

    acc_history = [[] for _ in range(epochs)]

    for ite in range(ite_num):
        print(f"iteration {ite+1}/ {ite_num}")
        set_seed(ite)

        if model_name == "AlexNet":
            model = AlexNet().to(device)
        elif model_name == "resnet18":
            model = resnet18(**params).to(device)
        elif model_name == "resnet34":
            model = resnet34(**params).to(device)
        elif model_name == "resnet50":
            model = resnet50(**params).to(device)
        elif model_name == "resnet101":
            model = resnet101(**params).to(device)
        elif model_name == "resnet152":
            model = resnet152(**params).to(device)
        elif model_name == "resnet20":
            model = resnet20(**params).to(device)
        elif model_name == "resnet32":
            model = resnet32(**params).to(device)
        elif model_name == "resnet44":
            model = resnet44(**params).to(device)
        elif model_name == "resnet56":
            model = resnet56(**params).to(device)
        elif model_name == "resnet110":
            model = resnet110(**params).to(device)
        elif model_name == "eenet18":
            model = eenet18(**params).to(device)
        elif model_name == "eenet34":
            model = eenet34(**params).to(device)
        elif model_name == "eenet50":
            model = eenet50(**params).to(device)
        elif model_name == "eenet101":
            model = eenet101(**params).to(device)
        elif model_name == "eenet152":
            model = eenet152(**params).to(device)
        elif model_name == "eenet20":
            model = eenet20(**params).to(device)
        elif model_name == "eenet32":
            model = eenet32(**params).to(device)
        elif model_name == "eenet44":
            model = eenet44(**params).to(device)
        elif model_name == "eenet56":
            model = eenet56(**params).to(device)
        elif model_name == "eenet110":
            model = eenet110(**params).to(device)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        optimizer = select_optimizer(optim_name, model, lr)

        # 100エポックごとに学習率を0.1倍するスケジューラー
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        model.train()
        for round in range(epochs):

            print(f"Round {round+1}/ {epochs}")

            train(optimizer, model, train_loader, device, model_name, loss_func, is_exit)

            scheduler.step()

            # モデルを評価
            if is_exit:
                test_thresholds = calculate_percentile_thresholds(model, val_loader, device, num_thresholds=exit_plot_num, num_ee=intermediate_exit_nums)
                test_acc = early_test(model, speed_test_loader, device, test_thresholds[-3], model_name, is_exit)
            else:
                # 通常モデルはしきい値計算をスキップし、Noneを渡す（early_test側が対応している場合）
                test_acc = early_test(model, speed_test_loader, device, None, model_name, is_exit)
            acc_history[round].append(test_acc)
            print(f"accuracy: {test_acc:.2f}%")
        
        # モデルの保存
        model_dir = os.path.join(parent_dir, "saved_models", f"{model_name}_{num_ee}_{ite+1}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    final_acc = acc_average(acc_history)

    save_dir = os.path.join(
        parent_dir,
        "results",
        "exp_03",
        f"{num_ee}exits_{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    rounds = np.arange(1, epochs + 1)

    # csv出力
    df = pd.DataFrame({
        "round": rounds,
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    plt.plot(rounds, final_acc, marker='o', linestyle='-', color='b', label='Average vs round')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, epochs + 1)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.png")

    plt.savefig(combined_path)
    