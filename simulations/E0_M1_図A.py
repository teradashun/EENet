# 図A：横軸 = round（1〜）、縦軸 = SI_conv ノルム（M1、対数軸）。
# 曲線2本（K=1, K=5）。期待：K=1 は平衡（≈57 付近で張り付く／既に観測済み）、K=5 は指数増大。


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

from src.utils import set_seed, choose_clients, select_optimizer, load_config, calculate_percentile_thresholds
from src.dataset import get_datasets, split_dataset
from src.local import train, test
from src.server import federated_learning
from src.flops_counter import get_model_complexity_info
from src.resnet import ResNet, ResNet6n2
from src.norm_tools import find_scale_invariant_conv_weights, capture_reference_norms, norm_report


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic Block defition.

    Basic 3X3 convolution blocks for use on ResNets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck Block defition.

    Bottleneck architecture for > 34 layer ResNets.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ExitBlock(nn.Module):
    """Exit Block defition.

    This allows the model to terminate early when it is confident for classification.
    """
    def __init__(self, inplanes, num_classes, input_shape, actual_inplanes=None):
        super(ExitBlock, self).__init__()
        _, width, height = input_shape

        self.adapter = None
        if actual_inplanes is not None and actual_inplanes != inplanes:
            self.adapter = nn.Conv2d(actual_inplanes, inplanes, kernel_size=1, bias=False)

        hidden_channels = max(inplanes * 2, 64)
        self.expansion = 1

        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.Conv2d(inplanes, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        ])

        self.layers.append(nn.AdaptiveAvgPool2d(1))

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * self.expansion, num_classes),
        )

    def forward(self, x):
        if self.adapter is not None:
            x = self.adapter(x)

        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)
        return pred


class EENet(nn.Module):
    """Builds a EENet like architecture.

    Arguments are
    * is_6n2model:        Whether the architecture of the model is 6n+2 layered ResNet.
    * block:              Block function of the architecture either 'BasicBlock' or 'Bottleneck'.
    * total_layers:       The total number of layers.
    * repetitions:        Number of repetitions of various block units.
    * num_ee:             The number of early exit blocks.
    * distribution:       Distribution method of the early exit blocks.
    * num_classes:        The number of classes in the dataset.
    * zero_init_residual: Zero-initialize the last BN in each residual branch,
                          so that the residual branch starts with zeros,
                          and each residual block behaves like an identity. This improves the model
                          by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    * input_shape:        Input shape of the model according to dataset.

    Returns:
        The nn.Module.
    """
    def __init__(self, is_6n2model, block, total_layers, num_ee, distribution, num_classes,
                 input_shape, loss_func, repetitions=None, zero_init_residual=False,
                 **kwargs):
        super(EENet, self).__init__()
        self.is_6n2model = is_6n2model

        if is_6n2model:
            self.inplanes = 16
            repetitions = [(total_layers-2) // 6]*3
            counterpart_model = ResNet6n2(block, total_layers, num_classes, input_shape)
        else:
            self.inplanes = 64
            counterpart_model = ResNet(block, repetitions, num_classes, input_shape)

        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.complexity = []
        self.cost = []
        self.layers = nn.ModuleList()
        self.stage_id = 0
        self.num_ee = num_ee
        self.total_layers = total_layers
        self.distribution = distribution
        self.num_classes = num_classes
        self.input_shape = input_shape

        channel, _, _ = input_shape
        total_flops, total_params = self.get_complexity(counterpart_model)
        self.set_thresholds(distribution, total_flops)

        if is_6n2model:
            self.layers.append(nn.Sequential(
                nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ))
        else:
            self.layers.append(nn.Sequential(
                nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))

        planes = self.inplanes
        stride = 1
        for repetition in repetitions:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            self.layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            if self.is_suitable_for_exit():
                self.add_exit_block(total_flops)

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))
                if self.is_suitable_for_exit():
                    self.add_exit_block(total_flops)

            planes *= 2
            stride = 2

        assert len(self.exits) == num_ee, \
            'The desired number of exit blocks is too much for the model capacity.'

        planes = 64 if is_6n2model else 512

        self.final_adapter = None
        if is_6n2model:
            num_exit = num_ee + 1
            final_inplanes = self.get_exit_inplanes(num_exit, num_ee)
            if final_inplanes != planes:
                self.final_adapter = nn.Conv2d(planes, final_inplanes, kernel_size=1, bias=False)
                planes = final_inplanes
                
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
            nn.Linear(planes * block.expansion, num_classes),
        )
        self.stages.append(nn.Sequential(*self.layers))
        self.softmax = nn.Softmax(dim=1)
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer(zero_init_residual)


    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(model, self.input_shape,\
                        print_per_layer_stat=False, as_strings=False)
        return flops, params
    

    def get_exit_inplanes(self, num_exit, idx):
        if num_exit == 1:
            return 32
        elif num_exit == 2:
            if idx == 0: return 16
            else: return 32
        elif 3 <= num_exit <= 5:
            if idx == 0: return 8
            elif idx in [1, 2]: return 16
            else: return 32
        elif 6 <= num_exit <= 8:
            if idx == 0: return 8
            elif idx in [1, 2, 3]: return 16
            else: return 32
        elif num_exit >= 9:
            if idx in [0, 1]: return 8
            elif 2 <= idx <= 5: return 16
            else: return 32
        return 64


    def add_exit_block(self, total_flops):
        """add early-exit blocks to the model

        Argument is
        * total_flops:   the total FLOPs of the counterpart model.

        This add exit blocks to suitable intermediate position in the model,
        and calculates the FLOPs and parameters until that exit block.
        These complexity values are saved in the self.cost and self.complexity.
        """
        self.stages.append(nn.Sequential(*self.layers))

        inplanes = self.inplanes
        actual_inplanes = None
        if getattr(self, 'is_6n2model', False):
            num_exit = self.num_ee + 1
            idx = len(self.exits)
            inplanes = self.get_exit_inplanes(num_exit, idx)
            actual_inplanes = self.inplanes

        self.exits.append(ExitBlock(inplanes, self.num_classes, self.input_shape, actual_inplanes=actual_inplanes))
        intermediate_model = nn.Sequential(*(list(self.stages)+list(self.exits)[-1:]))
        flops, params = self.get_complexity(intermediate_model)
        self.cost.append(flops / total_flops)
        self.complexity.append((flops, params))
        self.layers = nn.ModuleList()
        self.stage_id += 1


    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee+1)
        self.threshold = []
        for i in range(self.num_ee):
            if distribution == 'pareto':
                self.threshold.append(total_flops * (1 - (0.8**(i+1))))
            elif distribution == 'fine':
                self.threshold.append(total_flops * (1 - (0.95**(i+1))))
            elif distribution == 'linear':
                self.threshold.append(total_flops * flop_margin * (i+1))
            else:
                self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))


    def is_suitable_for_exit(self):
        """is the position suitable to locate an early-exit block"""
        intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
        flops, _ = self.get_complexity(intermediate_model)
        return self.stage_id < self.num_ee and flops >= self.threshold[self.stage_id]


    def parameter_initializer(self, zero_init_residual):
        """
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros,
        and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)


    def forward(self, x, exit_threshold=None):
        preds, confs = [], []

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            pred = exitblock(x)

            if not self.training and exit_threshold is not None:
                probs = F.softmax(pred, dim=1)
                max_probs, _ = torch.max(probs, dim=1)

                if max_probs[0] > exit_threshold[idx]:
                    return pred, idx

            preds.append(pred)

        x = self.stages[-1](x)

        if getattr(self, 'final_adapter', None) is not None:
            x = self.final_adapter(x)
        
        x = x.view(x.size(0), -1)
        pred = self.fc(x)

        if not self.training and exit_threshold is not None:
            return pred, len(self.exits)

        preds.append(pred)
        return preds


def eenet20(**kwargs):
    """EENet-20 model"""
    model = EENet(True, BasicBlock, 20, **kwargs)
    return model

def eenet32(**kwargs):
    """EENet-32 model"""
    model = EENet(True, BasicBlock, 32, **kwargs)
    return model

def eenet44(**kwargs):
    """EENet-44 model"""
    model = EENet(True, BasicBlock, 44, **kwargs)
    return model

def eenet56(**kwargs):
    """EENet-56 model"""
    model = EENet(True, BasicBlock, 56, **kwargs)
    return model

def eenet110(**kwargs):
    """EENet-110 model"""
    model = EENet(True, BasicBlock, 110, **kwargs)
    return model


def build_global_model(model_name, params, device):
    if model_name == "AlexNet":
        model = AlexNet().to(device)
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

    model = convert_bn_to_gn(model).to(device)
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
            outputs = model(inputs, exit_threshold=None) 
            
            if not corrects:
                corrects = [0] * len(outputs)
            
            for i, out in enumerate(outputs):
                _, predicted = out.max(1)
                corrects[i] += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
    return [100.0 * c / total for c in corrects]

"""
@torch.no_grad()
def bn_axis_check(model, eval_loader, calib_loader, device):
    def _eval(m):
        corrects, total = [], 0
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = m(inputs, exit_threshold=None)   # 全Exitのリストが返る
            if not corrects:
                corrects = [0] * len(outputs)
            for i, out in enumerate(outputs):
                _, pred = out.max(1)
                corrects[i] += pred.eq(targets).sum().item()
            total += targets.size(0)
        return [round(100.0 * c / total, 2) for c in corrects]

    # (A) 現行: 凍結された平均running stats
    model.eval()
    a = _eval(model)

    # (B) 診断: バッチ統計を使う(BNだけtrainモード)
    model.train()
    b = _eval(model)

    # (C) 修正候補: running statsをリセットして校正データで再計算 → eval
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    for i, (x, _) in enumerate(calib_loader):   # test以外のデータを使う
        model(x.to(device))
        if i + 1 >= 50:
            break
    model.eval()
    c = _eval(model)

    print(f"(A) eval凍結統計       : {a}")
    print(f"(B) trainモードBN(診断): {b}")
    print(f"(C) BN再計算後eval     : {c}")
"""


def convert_bn_to_gn(module, num_groups=8, affine=False):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = num_groups if c % num_groups == 0 else 1
            setattr(module, name, nn.GroupNorm(g, c, affine=affine))
        else:
            convert_bn_to_gn(child, num_groups, affine)
    return module


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
    num_ee       = config['training']['num_ee']
    distribution  = config['training']['distribution']
    loss_func     = config['training']['loss_func']
    zero_init_residual = config['training']['zero_init_residual']

    global_rounds = config['FL']['global_rounds']
    num_clients   = config['FL']['num_clients']
    dirichlet     = config['FL']['dirichlet']
    ite_num       = config['FL']['ite_num']
    cohort        = config['FL']['cohort']

    is_exit_model = ["eenet18", "eenet34", "eenet50", "eenet101", "eenet152", "eenet20", "eenet32", "eenet44", "eenet56", "eenet110"]
    if model_name in is_exit_model:
        is_exit = True
    else:
        is_exit = False
    
    # 最終Exit分を含めない
    intermediate_exit_nums = num_ee - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, _, val_loader, test_loader = get_datasets(batch_size, dataset_name, val_ratio=0.2)
    subsets = split_dataset(train_dataset, num_clients, dirichlet)

    # 各クライアントのデータローダーを作成
    client_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True)
                       for subset in subsets]

    num_classes = 10 if dataset_name in ["MNIST", "CIFAR10"] else 100
    input_shape = (1, 28, 28) if dataset_name == "MNIST" else (3, 32, 32)

    params = {
        "num_ee": intermediate_exit_nums,
        "distribution": distribution,
        "loss_func": loss_func,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "zero_init_residual": zero_init_residual,
    }

    acc_history = []
    norm_rows = []

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")
        set_seed(ite)

        global_model = build_global_model(model_name, params, device)
        si_names = find_scale_invariant_conv_weights(global_model)
        ref_norms = capture_reference_norms(global_model, si_names)

        ite_acc = []

        for round_idx in range(global_rounds):

            print(f"Round {round_idx+1}/ {global_rounds}")

            client_updates = []

            for client_idx in choose_clients(num_clients, cohort):
                local_model = copy.deepcopy(global_model)
                local_optimizer = select_optimizer(
                    optim_name,
                    local_model,
                    lr
                    )

                # ローカルでモデルを訓練
                for _ in range(epochs):
                    train(local_optimizer, local_model, client_loaders[client_idx], device, model_name, loss_func,is_exit)

                client_updates.append(local_model.state_dict())

            # クライアントのモデルを平均化してグローバルモデルを更新
            global_model = federated_learning(client_updates, global_model)
            # global_model = federated_learning(client_updates, global_model, si_names, ref_norms)

            rep = norm_report(global_model, si_names)
            norm_rows.append({"ite": ite, "round": round_idx + 1, **rep}) 

            test_accs = evaluate_all_exits(global_model, test_loader, device)
            ite_acc.append(test_accs)

            formatted_accs = [f"{acc:.2f}" for acc in test_accs]
            print(f"Test Accuracies for all exits: {formatted_accs}")
        
        acc_history.append(ite_acc)
    
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=0)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "E0",
        "M1",
        f"K={num_clients}"
        )
    os.makedirs(save_dir, exist_ok=True)
    
    # csv出力
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(os.path.join(save_dir, "norm_report.csv"), index=False)

    norm_mean = norm_df.groupby("round").mean().drop(columns=["ite"]).reset_index()
    norm_mean.to_csv(os.path.join(save_dir, "norm_report_mean.csv"), index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))
    plt.plot(norm_mean["round"], norm_mean["SI_conv"], marker="o",
         label=f"SI_conv (K={num_clients})")
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("SI_conv weight norm (log)")
    plt.title("SI_conv weight norm over rounds")
    plt.grid(True)
    plt.legend() # 凡例
    plt.savefig(os.path.join(save_dir, "SIconv.png"))