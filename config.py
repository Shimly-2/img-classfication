# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'classifier'  # visdom 环境
    vis_port =8097 # visdom 端口

    # 数据集处理相关
    Pro = [8, 1, 1]    # 数据集划分比例 train：val：test
    data_orgin = './data/train/'  # 原始数据集存放路径
    data_new = './datanew/'  # 划分后数据集存放路径

    # 数据增强相关
    beta = 1.0
    mean = [0.4020, 0.3766, 0.3820]  # for traffic image
    std = [0.2101, 0.2275, 0.2301]
    # mean=[0.485, 0.456, 0.406]  # for imagenet
    # std=[0.229, 0.224, 0.225]

    # 模型训练相关
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    # SqueezeNet/AlexNet/ResNet34/ResNet101/ResNest50
    num_classes = 62 # 分类数目

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test/'  # 测试集存放路径
    val_data_root = './data/val/'
    load_model_path = None           # 加载预训练的模型的路径，为None代表不加载
    best_model_path = './checkpoints/best_model.pth'

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    use_checkpoints = False  # use checkpoints or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 50
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    optimizer = 'Ranger'  # Adam/SGD/Ranger

    # 测试相关
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        print('-------------------------------------------------------------------')
        print('==> Printing user config..')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km='['+str(k)+']'
                print('{:<20}{:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位
        print('-------------------------------------------------------------------')

    def _parsehelp(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km = '[' + str(k) + ']'
                print('            {:<18}--- {:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位



opt = DefaultConfig()
