import copy
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.eenet import eenet18, eenet34, eenet50, eenet101, eenet152, eenet20, eenet32, eenet44, eenet56, eenet110
from src.utils import set_seed, choose_clients, select_optimizer, load_config, acc_average, calculate_percentile_thresholds
from src.dataset import get_datasets, split_dataset
from src.trainer import train, test
from src.server import federated_learning

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # FedAvg/src
    parent_dir = os.path.dirname(current_dir)                  # FedAvg
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size    = config['training']['batch_size']
    epochs        = config['training']['epochs']
    lr            = config['training']['lr']
    optimizer     = config['training']['optimizer']
    model_name    = config['training']['model_name']
    dataset_name  = config['training']['dataset_name']
    early_exit    = config['training']['early_exit']


    global_rounds = config['FL']['global_rounds']
    num_clients   = config['FL']['num_clients']
    dirichlet     = config['FL']['dirichlet']
    ite_num       = config['FL']['ite_num']
    cohort        = config['FL']['cohort']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_loader = get_datasets(batch_size, dataset_name)
    subsets = split_dataset(train_dataset, num_clients, dirichlet)

    # 各クライアントのデータローダーを作成
    client_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True)
                       for subset in subsets]

    if dataset_name == "MNIST":
        if early_exit:
            global_model = B_LeNet_5().to(device)
        
        else:
            global_model = MNIST_CNN().to(device)

    elif dataset_name == "CIFAR_10":
        global_model = CIFAR10_CNN().to(device)
    
        
    global_model.train()

    acc_history = [[] for _ in range(global_rounds)]

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")

        set_seed(ite)

        for round in range(global_rounds):

            print(f"Round {round+1}/ {global_rounds}")

            client_updates = []

            for client_idx in choose_clients(num_clients, cohort):
                local_model = copy.deepcopy(global_model)
                local_optimizer = select_optimizer(
                    optimizer,
                    local_model,
                    lr
                    )

                # ローカルでモデルを訓練
                for _ in range(epochs):
                    train(local_optimizer, local_model, client_loaders[client_idx], device, model_name)

                client_updates.append(local_model.state_dict())

            # クライアントのモデルを平均化してグローバルモデルを更新
            federated_learning(client_updates, global_model)

            # グローバルモデルを評価
            test_acc = test(global_model, test_loader, device)
            acc_history[round].append(test_acc)
    
    final_acc = acc_average(acc_history)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        f"{dataset_name}"
        f"{model_name}"
        f"{dirichlet},{num_clients},{dirichlet}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "round": global_rounds,
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, "FedAvg.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    rounds = np.arange(1, global_rounds + 1) # グラフのX軸（ラウンド数）

    plt.plot(rounds, final_acc, linestyle='-', color='b', label='Simple Average')

    # グラフのタイトルとラベル
    plt.title('FedAvg', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, global_rounds)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, "FedAvg.png")

    plt.savefig(combined_path)
        