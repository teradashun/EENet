import torch
import torch.nn as nn
import torch.nn.functional as F


class B_LeNet_5(nn.Module):
    def __init__(self):
        super(B_LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1,2)  # (input_channels, output_channels, Kernel_size, stride)
        self.pool = nn.AvgPool2d(2,2)  # (Kernel_size, stride)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.conv3 = nn.Conv2d(16,120,5,1)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.b_conv = nn.Conv2d(6,16,3,1)
        self.b_fc = nn.Linear(16*6*6,10)
    
    def forward(self, x, threshold=None):
        x = self.pool(F.tanh(self.conv1(x)))

        # 分岐1
        x_branch = self.pool(F.tanh(self.b_conv(x)))
        x_branch = x_branch.view(-1, 16*6*6)
        out_branch = self.b_fc(x_branch)
        if threshold is not None:
            probs_branch = F.softmax(out_branch, dim=1)
            entropy = torch.sum(-probs_branch * torch.log(probs_branch + 1e-9), dim=1)
            if torch.all(entropy <= threshold):
                return out_branch, None

        # main
        # (14x14x6 -> 10x10x16 -> 5x5x16)
        x = self.pool(F.tanh(self.conv2(x)))
        # (5x5x16 -> 1x1x120)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        # (1x1x120 -> 1x1x84)
        x = F.tanh(self.fc1(x))
        # (1x1x84 -> 1x1x10)
        out_main = self.fc2(x)
        return out_branch, out_main


class B_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力: 16x16
            )
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, num_classes),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 8x8
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, num_classes)
        )
        
        self.layer3_5 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 出力: 4x4
        )

        #  Main Classifier
        self.branch3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor, threshold=None) -> torch.Tensor:
        x1 = self.layer1(x)
        out1 = self.branch1(x1)
        if threshold is not None:
            probs_branch1 = F.softmax(out1, dim=1)
            entropy1 = torch.sum(-probs_branch1 * torch.log(probs_branch1 + 1e-9), dim=1)
            if torch.all(entropy1 <= threshold[0]):
                return out1, None, None

        x2 = self.layer2(x1)
        out2 = self.branch2(x2)
        if threshold is not None:
            probs_branch2 = F.softmax(out2, dim=1)
            entropy2 = torch.sum(-probs_branch2 * torch.log(probs_branch2 + 1e-9), dim=1)
            if torch.all(entropy2 <= threshold[1]):
                return out1, out2, None

        x3 = self.layer3_5(x2)
        x3 = torch.flatten(x3, 1)
        out3 = self.branch3(x3)

        return out1, out2, out3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class B_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(B_ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Main Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Early Exits
        self.exit1 = self._make_early_exit(64 * block.expansion, num_classes)
        self.exit2 = self._make_early_exit(128 * block.expansion, num_classes)
        self.exit3 = self._make_early_exit(256 * block.expansion, num_classes)

        # Main Exit (Exit 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _make_early_exit(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x, threshold=None):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        out1 = self.exit1(x)
        if threshold is not None:
            probs_branch1 = F.softmax(out1, dim=1)
            entropy1 = torch.sum(-probs_branch1 * torch.log(probs_branch1 + 1e-9), dim=1)
            if torch.all(entropy1 <= threshold[0]):
                return out1, None, None, None

        x = self.layer2(x)
        out2 = self.exit2(x)
        if threshold is not None:
            probs_branch2 = F.softmax(out2, dim=1)
            entropy2 = torch.sum(-probs_branch2 * torch.log(probs_branch2 + 1e-9), dim=1)
            if torch.all(entropy2 <= threshold[1]):
                return out1, out2, None, None

        x = self.layer3(x)
        out3 = self.exit3(x)
        if threshold is not None:
            probs_branch3 = F.softmax(out3, dim=1)
            entropy3 = torch.sum(-probs_branch3 * torch.log(probs_branch3 + 1e-9), dim=1)
            if torch.all(entropy3 <= threshold[2]):
                return out1, out2, out3, None

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out4 = self.fc(x)

        return out1, out2, out3, out4


def B_ResNet_18(num_classes=100):
    return B_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def B_ResNet_34(num_classes=100):
    return B_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)