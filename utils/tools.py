import os
import torch as t
import models
from torchvision import datasets
from torchvision import transforms as T
import numpy as np

def get_random_images(num, mean, std, test_data_root):
    # 数据转换操作，测试验证和训练的数据转换有所区别
    normalize = T.Normalize(mean=mean, std=std)
    # 测试集和验证集
    transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224,scale=(0.75,1.25),ratio=(0.6,1.0)),   # 随机大小、长宽比裁剪
                T.RandomHorizontalFlip(p=0.5),   # 依概率p水平翻转
                T.RandomVerticalFlip(p=0.5),     # 依概率p垂直翻转
                # T.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),    # 随机亮度
                # T.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),    # 随机对比度
                T.ToTensor(),
                normalize])
    data = datasets.ImageFolder(test_data_root, transform=transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = t.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

def predict_image(image, model, best_model_path, device):
    # configure model 模型
    model = getattr(models, model)().eval()
    if best_model_path:
        model.load(best_model_path)
    model.to(device)
    test_transforms=T.Compose([T.ToTensor()])
    image_tensor = test_transforms(image).float()   # 将图片转为Tensor
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    input = input.to(device)
    output = model(input)
    label = output.max(dim = 1)[1].detach().tolist()
    return label


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)