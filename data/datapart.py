import os, random, shutil

# 用于划分训练集，验证集以及测试集

def make_dir(source, target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    dir_names = os.listdir(source)
    for names in dir_names:
        for i in ['train', 'val', 'test']:
            path = target + '/' + i + '/' + names
            if not os.path.exists(path):
                os.makedirs(path)

def make_dir_notest(source, target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    dir_names = os.listdir(source)
    for names in dir_names:
        for i in ['train', 'val']:
            path = target + '/' + i + '/' + names
            if not os.path.exists(path):
                os.makedirs(path)


def divideTrainValiTest(source, target, Pro):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''
    # 得到源文件下的种类
    make_dir(source, target)
    pic_name = os.listdir(source)
    trainnum = 0
    valnum = 0
    testnum = 0

    print('-------------------------------------------------------------------')
    print('==> Start data divide..')

    # 对于每一类里的数据进行操作
    for classes in pic_name:
        # 得到这一种类的图片的名字
        pic_classes_name = os.listdir(os.path.join(source, classes))
        random.shuffle(pic_classes_name)

        # 按照8：1：1比例划分
        train_list = pic_classes_name[0:int(Pro[0] * 0.1 * len(pic_classes_name))]
        valid_list = pic_classes_name[int(Pro[0] * 0.1 * len(pic_classes_name)):int((Pro[0] + Pro[1]) * 0.1 * len(pic_classes_name))]
        test_list = pic_classes_name[int((Pro[0] + Pro[1]) * 0.1 * len(pic_classes_name)):]

        # 对于每个图片，移入到对应的文件夹里面
        for train_pic in train_list:
            shutil.copyfile(source + '/' + classes + '/' + train_pic, target + '/train/' + classes + '/' + train_pic)
            trainnum = trainnum + 1
        for validation_pic in valid_list:
            shutil.copyfile(source + '/' + classes + '/' + validation_pic, target + '/val/' + classes + '/' + validation_pic)
            valnum = valnum + 1
        for test_pic in test_list:
            shutil.copyfile(source + '/' + classes + '/' + test_pic, target + '/test/' + classes + '/' + test_pic)
            testnum = testnum + 1
        km = '[' + str(classes) + ']'
        print('{:<7}{:<8}'.format('Class', str(km)), 'finished')

    print('Total number of images: ', trainnum + valnum + testnum)
    print('Number of Training images: ', trainnum)
    print('Number of Validation images: ', valnum)
    print('Number of Testing images: ', testnum)
    print('-------------------------------------------------------------------')


def divideTrainVali(source, target, Pro):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''
    # 得到源文件下的种类
    make_dir_notest(source, target)
    pic_name = os.listdir(source)
    trainnum = 0
    valnum = 0

    print('-------------------------------------------------------------------')
    print('==> Start data divide..')

    # 对于每一类里的数据进行操作
    for classes in pic_name:
        # 得到这一种类的图片的名字
        pic_classes_name = os.listdir(os.path.join(source, classes))
        random.shuffle(pic_classes_name)

        # 按照8：2：0比例划分
        train_list = pic_classes_name[0:int(Pro[0] * 0.1 * len(pic_classes_name))]
        valid_list = pic_classes_name[int(Pro[0] * 0.1 * len(pic_classes_name)):]

        # 对于每个图片，移入到对应的文件夹里面
        for train_pic in train_list:
            shutil.copyfile(source + '/' + classes + '/' + train_pic, target + '/train/' + classes + '/' + train_pic)
            trainnum = trainnum + 1
        for validation_pic in valid_list:
            shutil.copyfile(source + '/' + classes + '/' + validation_pic, target + '/val/' + classes + '/' + validation_pic)
            valnum = valnum + 1
        km = '[' + str(classes) + ']'
        print('{:<7}{:<8}'.format('Class',str(km)),'finished')

    print('Total number of images: ',trainnum + valnum)
    print('Number of Training images: ', trainnum)
    print('Number of Validation images: ', valnum)
    print('-------------------------------------------------------------------')

# if __name__ == '__main__':
#     filepath = r'../data'
#     dist = r'../datanew'
#     make_dir(filepath, dist)
#     divideTrainValiTest(filepath, dist)
#     print("divide success")
