import torch
from torch import nn
import torch.nn.functional as F
from src.flops_counter import get_model_complexity_info
from src.resnet import ResNet, ResNet6n2

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

        hidden_channels = max(inplanes * 2, 256)
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
            final_inplanes = self.get_exit_inplanes(num_exit, total_flops)
            if final_inplanes != planes:
                self.final_adapter = nn.Conv2d(planes, final_inplanes, kernel_size=1, bias=False)
                planes = final_inplanes
                
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        #self.fully_connected = nn.Linear(planes * block.expansion, num_classes)
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
            if idx == 0: return 32
            else: return 64
        elif 3 <= num_exit <= 5:
            if idx == 0: return 16
            elif idx in [1, 2]: return 32
            else: return 64
        elif 6 <= num_exit <= 8:
            if idx == 0: return 16
            elif idx in [1, 2, 3]: return 32
            else: return 64
        elif num_exit >= 9:
            if idx in [0, 1]: return 16
            elif 2 <= idx <= 5: return 32
            else: return 64
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
            if not self.training:
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
        if not self.training:
            return pred, len(self.exits)
        preds.append(pred)
        return preds


def eenet18(**kwargs):
    """EENet-18 model"""
    model = EENet(is_6n2model=False, block=BasicBlock, total_layers=18,
                  repetitions=[2, 2, 2, 2], **kwargs)
    return model

def eenet34(**kwargs):
    """EENet-34 model"""
    model = EENet(is_6n2model=False, block=BasicBlock, total_layers=34,
                  repetitions=[3, 4, 6, 3], **kwargs)
    return model

def eenet50(**kwargs):
    """EENet-50 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=50,
                  repetitions=[3, 4, 6, 3], **kwargs)
    return model

def eenet101(**kwargs):
    """EENet-101 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=101,
                  repetitions=[3, 4, 23, 3], **kwargs)
    return model

def eenet152(**kwargs):
    """EENet-152 model"""
    model = EENet(is_6n2model=False, block=Bottleneck, total_layers=152,
                  repetitions=[3, 8, 36, 3], **kwargs)
    return model

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