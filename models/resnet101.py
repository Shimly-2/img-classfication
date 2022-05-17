from torchvision.models import  squeezenet1_1
from models.basic_module import  BasicModule
from torch import nn
import torch as t
from torch.optim import Adam, SGD
from config import opt
from torchvision import models
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py

# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = t.mean(x, dim=1, keepdim=True)
        max_out, _ = t.max(x, dim=1, keepdim=True)
        x = t.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet101(BasicModule):
    def __init__(self, num_classes=opt.num_classes):
        super(ResNet101, self).__init__()
        self.model_name = 'resnet101'
        self.model = models.resnet101(pretrained=True)
        # 对于模型的每个权重，使其不进行反向传播，即固定参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
        for param in self.model.fc.parameters():
            param.requires_grad = True
        # 修改 原始的num_class: 预训练模型是1000分类
        channel_in = self.model.fc.in_features  # 获取fc层的输入通道数
        # 然后把resnet-101的fc层替换成300类别的fc层
        self.model.fc = nn.Linear(channel_in, num_classes)
        # self.ca = ChannelAttention(num_classes)
        # self.sa = SpatialAttention()

    def forward(self,x):
        # self.ca = ChannelAttention(x) * x
        # self.sa = SpatialAttention(x) * x
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        # 因为使用了预训练模型，我们只需要训练后面的分类
        # 前面的特征提取部分可以保持不变
        if opt.optimizer == 'Adam':
            return t.optim.Adam(self.fc.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'SGD':
            return t.optim.SGD(self.fc.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'Ranger':
            return Ranger(self.fc.parameters(), lr=lr, weight_decay=weight_decay)