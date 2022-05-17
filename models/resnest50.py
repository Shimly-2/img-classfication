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
from resnest.torch import resnest50


class ResNest50(BasicModule):
    def __init__(self, num_classes=opt.num_classes):
        super(ResNest50, self).__init__()
        self.model_name = 'resnest50'
        self.model = resnest50(pretrained=True)
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

    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        # 因为使用了预训练模型，我们只需要训练后面的分类
        # 前面的特征提取部分可以保持不变
        if opt.optimizer == 'Adam':
            return t.optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'SGD':
            return t.optim.SGD(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'Ranger':
            return Ranger(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
