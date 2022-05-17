#coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat, UnNormalize
from data.datapart import divideTrainValiTest,divideTrainVali
from data.dataAnalysis import Statisticsdata, cal
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler
from utils.tools import *

# 利用随机数种子来使pytorch中的结果可以复现
SEED = 15
t.manual_seed(SEED)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
np.random.seed(SEED)

# 数据转换操作，测试验证和训练的数据转换有所区别
normalize = T.Normalize(mean=opt.mean,std=opt.std)
# 训练集数据增强
train_transforms = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224,scale=(0.75,1.25),ratio=(0.6,1.0)),   # 随机大小、长宽比裁剪
            T.RandomHorizontalFlip(p=0.5),   # 依概率p水平翻转
            T.RandomVerticalFlip(p=0.5),     # 依概率p垂直翻转
            # T.ColorJitter(brightness=0.4, contrast=0, saturation=0, hue=0),    # 随机亮度
            # T.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0),    # 随机对比度
            T.ToTensor(),
            normalize])
# 验证集和测试集数据增强
test_transforms = T.Compose([
            T.Resize((224, 224)),
            # T.RandomResizedCrop(224),   # 随机大小、长宽比裁剪
            T.ToTensor(),
            normalize])

# 用于数据集分析
def dataanalysis():
    Statisticsdata(opt.data_orgin)       # 分析数据集每类图像占比
    cal(opt.data_orgin,opt.num_workers)  # 分析数据集图像的平均值和标准差

# 用于数据集划分
def datadivide():
    if(opt.Pro[2] == 0):
        divideTrainVali(opt.data_orgin,opt.data_new, opt.Pro)        # 不存在测试集时，只划分train、val
    else:
        divideTrainValiTest(opt.data_orgin, opt.data_new, opt.Pro)   # 存在测试集时，划分train、val、test

@t.no_grad() # pytorch>=0.5  抽取随机图片可视化验证正确率
def testforimg(**kwargs):
    opt._parse(kwargs)

    # configure model 模型
    model = getattr(models, opt.model)().eval()
    if opt.best_model_path:
        model.load(opt.best_model_path)
    model.to(opt.device)

    print('-------------------------------------------------------------------')
    print('==> Start test for random images..')

    test_datasets = datasets.ImageFolder(opt.test_data_root, transform=test_transforms)
    classes = test_datasets.classes
    to_pil = T.ToPILImage()
    images, labels = get_random_images(10, opt.mean, opt.std, opt.test_data_root)
    # images, labels = get_random_images(10)   # 获取随机图片
    fig = plt.figure(figsize=(10, 10))
    unorm = UnNormalize(mean=opt.mean, std=opt.std) # 实例化反归一器
    for ii in range(len(images)):
        temp = unorm(images[ii])   # 反归一化
        image = to_pil(temp)
        image = to_pil(images[ii])
        index = predict_image(image, opt.model, opt.best_model_path, opt.device)
        # temp = unorm(images[ii])  # 反归一化
        # image = to_pil(temp)
        sub = fig.add_subplot(1, len(images), ii + 1)
        res = int(labels[ii]) == index[0]
        print('[Result]:','Origin Label',int(labels[ii]),', Predict Label',index[0])
        sub.set_title(str(classes[index[0]]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()
    print('-------------------------------------------------------------------')

@t.no_grad() # pytorch>=0.5  将验证结果写入csv中
def testforcsv(**kwargs):
    opt._parse(kwargs)

    # configure model 模型
    model = getattr(models, opt.model)().eval()
    if opt.best_model_path:
        model.load(opt.best_model_path)
    model.to(opt.device)

    print('-------------------------------------------------------------------')
    print('==> Start test for write csv..')
    # data 数据
    test_datasets = datasets.ImageFolder(opt.test_data_root, transform=test_transforms)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score,dim=1)[:,0].detach().tolist()
        label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(),probability_) for path_,probability_ in zip(path,label) ]

        results += batch_results
    write_csv(results,opt.result_file)
    print('Csv write successfully')
    print('-------------------------------------------------------------------')

@t.no_grad() # pytorch>=0.5 计算测试集上的正确率
def testforacc(**kwargs):
    opt._parse(kwargs)

    # configure model 模型
    model = getattr(models, opt.model)().eval()
    if opt.best_model_path:
        model.load(opt.best_model_path)
    model.to(opt.device)

    acc = 0
    test_datasets = datasets.ImageFolder(opt.test_data_root, transform=test_transforms)
    classes = test_datasets.classes
    test_dataloader = t.utils.data.DataLoader(test_datasets,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    print('-------------------------------------------------------------------')
    print('==> Start test for total accuracy..')
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        # print(path,ii)
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score,dim=1)[:,0].detach().tolist()
        label = score.max(dim = 1)[1].detach().tolist()
        for i in range(len(path)):
            res = int(path[i]) == label[i]
            acc += res
        # batch_results = [(path_.item(),probability_) for path_,probability_ in zip(path,probability) ]
        #
        # results += batch_results
    acc = acc/(len(test_dataloader)*opt.batch_size)
    print('The test accuracy is:', 100*acc, '%')
    print('-------------------------------------------------------------------')
    # print(results)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data 数据增强

    # step3: 数据平衡，权重采样
    # 统计权重
    pic_name = os.listdir(opt.train_data_root)
    picnum = []
    for classes in pic_name:
        # 得到这一种类的图片的名字
        pic_classes_name = os.listdir(os.path.join(opt.train_data_root, classes))
        picnum.append(len(pic_classes_name))
    weights = 1 / t.Tensor(picnum)
    sampler = t.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)  # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度。

    val_data = datasets.ImageFolder(opt.val_data_root,transform=test_transforms)
    train_data = datasets.ImageFolder(opt.train_data_root,transform=train_transforms)

    train_dataloader = DataLoader(train_data,
                                  batch_size = opt.batch_size,
                                  shuffle = False,
                                  sampler = ImbalancedDatasetSampler(train_data),
                                  num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_data,
                                batch_size = opt.batch_size,
                                shuffle = False,
                                num_workers = opt.num_workers)
    
    # step3: criterion and optimizer 损失函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    # optimizer = t.optim.Adam(model.parameters(),
    #                          lr=lr,
    #                          weight_decay=opt.weight_decay)
        
    # step4: meters 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    previous_loss = 1e10
    best_acc, best_epoch=0, 0

    # train 训练
    for epoch in range(opt.max_epoch):
        print('-------------------------------------------------------------------')
        print('==> Start training..')
        loss_meter.reset()
        confusion_matrix.reset()
        for ii, (data,label) in tqdm(enumerate(train_dataloader)):
            # train model 训练模型参数
            input = data.to(opt.device)
            target = label.to(opt.device)
            # 加入Cutmix 数据增强
            # if np.random.rand() < 0.5:
            #     # 清除当前所有的累积梯度
            #     optimizer.zero_grad()
            #     # generate mixed sample
            #     lam = np.random.beta(opt.beta, opt.beta)
            #     rand_index = t.randperm(input.size()[0]).cuda()
            #     target_a = target
            #     target_b = target[rand_index]
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            #     input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            #     # adjust lambda to exactly match pixel ratio
            #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            #     score = model(input)
            #     loss = criterion(score, target_a) * lam + criterion(score, target_b) * (1. - lam)
            #     # 反向传播
            #     loss.backward()
            #     # 修正模型参数
            #     optimizer.step()
            #     # meters update and visualize 更新统计指标以及可视化
            #     loss_meter.add(loss.item())
            #     # detach 一下更安全保险
            #     confusion_matrix.add(score.detach(), target.detach())
            # else:

            # 清除当前所有的累积梯度
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            # 反向传播
            loss.backward()
            # 修正模型参数
            optimizer.step()
            # meters update and visualize 更新统计指标以及可视化
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1)%opt.print_freq == 0:
                accuracy = 0
                cm_value = confusion_matrix.value()
                for i in range(opt.num_classes):
                    accuracy += 100. * cm_value[i][i] / (cm_value.sum())
                vis.plot('loss', loss_meter.value()[0])
                vis.plot('acc1', accuracy)
                vis.log("[TRAIN] Epoch={epoch}/{max_epoch}, Step={ii}/{img_num:.0f}, loss={loss:.6f}, acc1={acc1:.6f}, lr:{lr:.6f}"
                        .format(epoch=epoch+1,
                                max_epoch=opt.max_epoch,
                                loss=loss_meter.value()[0],
                                ii=ii+1,
                                acc1=accuracy,
                                img_num=len(train_dataloader),
                                lr=lr))
                
                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        if opt.use_checkpoints:
            model.save()
        vis.log("[TRAIN] Epoch={epoch} finished, loss={loss:.6f}"
                .format(epoch=epoch+1,
                        loss=loss_meter.value()[0]))
        vis.log("Start to evaluate(total_samples={total_samples}, total_steps={total_steps:.0f})..."
                .format(total_samples=len(val_dataloader)*opt.batch_size,
                        total_steps=len(val_dataloader)))
        # validate and visualize 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("[EVAL] Finished, Epoch={epoch}, acc1={val_acc}."
                .format(epoch=epoch+1,
                        val_acc=val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_epoch = epoch
            # 储存最好的模型结果
            t.save(model.state_dict(), opt.best_model_path)
        vis.log("Current evaluated best model on eval_dataset is epoch_{best_epoch}, acc1={best_acc}"
                .format(best_epoch=best_epoch+1,
                        best_acc=best_acc))
        
        # update learning rate 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

        previous_loss = loss_meter.value()[0]

@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """
    # 把模型设为验证模式
    print('-------------------------------------------------------------------')
    print('==> Start validation..')
    model.eval()
    vis = Visualizer(opt.env, port=opt.vis_port)
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
        if (ii + 1) % opt.print_freq == 0:
            vis.log("[EVAL] Step={ii}/{img_num:.0f}"
                    .format(ii=ii+1,
                            img_num=len(dataloader)))

    # 把模型恢复为训练模式
    model.train()
    accuracy = 0
    cm_value = confusion_matrix.value()
    for i in range(opt.num_classes):
        accuracy += 100. * cm_value[i][i] / (cm_value.sum())
    return confusion_matrix, accuracy

def help(**kwargs):
    """
    打印帮助的信息： python file.py help
    """
    print("""-------------------------------------------------------------------
    usage : python file.py <function> [--args=value]
    <function> := train | testforimg | testforacc | testforcsv | dataanalysis | datadivide | help 
            [train]        --- Start train model
            [testforimg]   --- Start test for random imgages
            [testforacc]   --- Start test for total acc
            [testforcsv]   --- Start test for write results to csv
            [dataanalysis] --- Start analysis for datasets
            [datadivide]   --- Start make divide for datasets
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} testforacc --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))
    opt._parsehelp(kwargs)

    # from inspect import getsource
    # source = (getsource(opt.__class__))
    # print(source)
    print('-------------------------------------------------------------------')


if __name__=='__main__':
    import fire
    fire.Fire()
