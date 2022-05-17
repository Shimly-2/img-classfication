#coding:utf8
import torch as t
import time
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py
from config import opt


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))   # 模型的默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path),strict=False)

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        如AlexNet_0710_23:57:29.pth
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        if opt.optimizer == 'Adam':
            return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'SGD':
            return t.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt.optimizer == 'Ranger':
            return Ranger(self.parameters(), lr=lr, weight_decay=weight_decay)




class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
