import os
import matplotlib.pyplot as plt
import torch as t
from torchvision import datasets
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.patheffects as PathEffects
import matplotlib
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def Statisticsdata(source):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
    '''
    # 得到源文件下的种类
    pic_name = os.listdir(source)

    picnum = []
    picname = []
    fig, ax = plt.subplots()

    print('-------------------------------------------------------------------')
    print('==> Printing data numbers for every classes..')
    # 对于每一类里的数据进行操作
    for classes in pic_name:
        # 得到这一种类的图片的名字
        pic_classes_name = os.listdir(os.path.join(source, classes))
        km = '[' + str(classes) + ']'
        print('{:<10}{:<5}'.format(str(km), str(len(pic_classes_name))),'images') # {:<30d}含义是 左对齐，且占用30个字符位
        picname.append(classes)
        picnum.append(len(pic_classes_name))
    print('-------------------------------------------------------------------')

    plt.bar(picname, picnum, label='data num')
    for a, b in zip(picname, picnum):
        ax.text(a, b + 1, b, ha='center', va='bottom')
    plt.xticks(rotation=90)  # 旋转90度
    plt.xlabel('classes')
    plt.ylabel('numbers')
    plt.title('data numbers for every classes')
    plt.show()

    # t-SNE 算法
    # TODO
    # RS = 20150101  # 随机状态值
    # X = np.vstack(picnum)
    # y = np.hstack(picname)
    # digits_proj = TSNE(random_state=RS).fit_transform(X)


# 计算自己数据集的标准差和平均值pyth
def cal(source,num_workers):
    N_CHANNELS = 3
    dataset = datasets.ImageFolder(source, transform=T.ToTensor())
    full_loader = t.utils.data.DataLoader(dataset, shuffle=False, num_workers=num_workers)

    mean = t.zeros(3)
    std = t.zeros(3)
    print('-------------------------------------------------------------------')
    print('==> Computing mean and std..')
    for inputs, _labels in tqdm(full_loader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print('The mean is:', mean, ', the std is:', std)
    print('-------------------------------------------------------------------')



