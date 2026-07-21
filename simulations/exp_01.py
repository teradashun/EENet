# 閾値 vs exit割合 (exit数4固定)

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

from simulations.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet20, resnet32, resnet44, resnet56, resnet110
from simulations.eenet import eenet18, eenet34, eenet50, eenet101, eenet152, eenet20, eenet32, eenet44, eenet56, eenet110
from simulations.utils import set_seed, load_config, select_optimizer, calculate_percentile_thresholds, society_thresholds
from simulations.loss_functions import loss
from simulations.flops_counter import get_model_complexity_info
from src.server import federated_learning
from src.dataset import get_datasets, split_dataset


def test(model, test_loader, device, model_name):
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for dummy_inputs, _ in test_loader:
            input_res = tuple(dummy_inputs.shape[1:])
            base_flops, _ = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
            break 

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            
            _, outputs = torch.max(pred, dim=1)

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds / total_preds
    
    exit_ratios = np.array([100.0])
    return acc, base_flops


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()

    num_exits = len(model.complexity)
    exit_counts = np.zeros(num_exits)

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0
        total_flops = 0

        for dummy_inputs, _ in test_loader:
            dummy_inputs = dummy_inputs.to(device)
            _ = model(dummy_inputs, threshold)
            break 

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred, exit_idx, cost = model(inputs, threshold)
            
            _, outputs = torch.max(pred, dim=1)

            sample_flops = model.complexity[exit_idx][0]
            total_flops += sample_flops

            exit_counts[exit_idx] += 1

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
        avg_flops = total_flops / total_preds

        exit_ratios = (exit_counts / total_preds) * 100
    
    return acc, avg_flops, exit_ratios


def run_inference(parent_dir, config, device, intermediate_exit_nums):
    batch_size          = config['training']['batch_size']
    dataset             = config['training']['dataset']
    num_ee              = config['training']['num_ee']
    distribution        = config['training']['distribution']
    exit_type           = config['training']['exit_type']
    loss_func           = config['training']['loss_func']
    zero_init_residual  = config['training']['zero_init_residual']
    ite_num             = config['FL']['ite_num']
    exit_plot_num       = config['training']['exit_plot_num']

    num_classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100
    input_shape = (1, 28, 28) if dataset == "MNIST" else (3, 32, 32)

    params = {
        "num_ee": intermediate_exit_nums,
        "distribution": distribution,
        "exit_type": exit_type,
        "loss_func": loss_func,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "zero_init_residual": zero_init_residual,
    }

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
    
    is_exit_model = ["eenet18", "eenet34", "eenet50", "eenet101", "eenet152", "eenet20", "eenet32", "eenet44", "eenet56", "eenet110"]
    if model_name in is_exit_model:
        is_exit = True
    else:
        is_exit = False

    _, val_loader, _ = get_datasets(batch_size, dataset, val_ratio=0.2)
    _, _, speed_test_loader = get_datasets(1, dataset, val_ratio=0.2)

    acc_history = []
    exit_dist_history = []
    thresholds_history = []

    for ite in range(ite_num):
        print(f"iteration {ite+1}/ {ite_num}")
        set_seed(ite)

        model_path = os.path.join(parent_dir, "saved_models", f"{model_name}_{num_ee}_{ite+1}", "model.pth")

        if os.path.exists(model_path):
            print(f"モデルをロードして推論を開始します。 モデル名: {model_name}, exit数: {num_ee}")
            print(model_path)
            model.load_state_dict(torch.load(model_path))

            if not is_exit:
                test_acc, _, exit_ratios = test(model, speed_test_loader, device, model_name)
                acc_history.append(test_acc)
                exit_dist_history.append(exit_ratios)
                thresholds_history.append(np.mean(th))

                print(f"閾値: {th}, accuracy: {test_acc:.2f}%")

            else:
                # test_thresholds = calculate_percentile_thresholds(model, val_loader, device, num_thresholds=exit_plot_num, num_ee=num_ee)
                test_thresholds = society_thresholds(model, val_loader, device, num_ee=intermediate_exit_nums, num_thresholds=exit_plot_num)

                iter_exit_dist = []
                iter_thresholds = []

                for th in test_thresholds:
                    test_acc, _, exit_ratios = early_test(model, speed_test_loader, device, th, model_name)
                    acc_history.append(test_acc)
                    iter_exit_dist.append(exit_ratios)
                    iter_thresholds.append(np.mean(th))
                    print(f"閾値: {np.mean(th):.4f}, accuracy: {test_acc:.2f}%")

                exit_dist_history.append(iter_exit_dist)
                thresholds_history.append(iter_thresholds)
    
    if not is_exit:
        final_thresholds = None
        final_exit_dist = None
    
    else:
        final_thresholds = np.array(thresholds_history).mean(axis=0)
        final_exit_dist = np.array(exit_dist_history).mean(axis=0)

    return final_thresholds, final_exit_dist


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # BranchyNet/simulations
    parent_dir = os.path.dirname(current_dir)                  # BranchyNet
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    model_name = config['training']['model_name']
    num_ee = config['training']['num_ee']
    ite_num = config['FL']['ite_num']

    # 最終Exit分を含めない
    intermediate_exit_nums = num_ee - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(parent_dir, "saved_models", f"{model_name}_{num_ee}_{ite_num+1}")
    
    final_thresholds, final_exit_dist = run_inference(parent_dir, config, device, intermediate_exit_nums)
    
    save_dir = os.path.join(
        parent_dir,
        "results",
        "exp_01",
        f"{num_ee}exits_{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "thresholds": final_thresholds,
    })

    for i in range(final_exit_dist.shape[1]):
        df[f"Exit_{i+1}"] = final_exit_dist[:, i]

    csv_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    for i in range(num_ee):
        plt.plot(final_thresholds, final_exit_dist[:, i], marker='o', linestyle='-', color='b', label=f'Exit {i+1}')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('Average Threshold', fontsize=12)
    plt.ylabel('Exit Ratio (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, 1.0)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.png")

    plt.savefig(combined_path)
    