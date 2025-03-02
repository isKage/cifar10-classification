import time

import os

import pandas as pd
import torch
import warnings


class DefaultConfig:
    user_root = os.path.join("D:\\")
    working_root = os.getcwd()  # 获取当前根目录

    '''
    "real": 真正开始训练和预测
    "try": 仅测试代码是否完整正确，用模拟数据测试
    '''
    real_or_try = "try"  # 默认
    # 样本数据尝试
    root = os.path.join(working_root, 'TempData', 'competitions', 'cifar-10')
    res_path = os.path.join(working_root, 'result_example.csv')

    model = "ResNet18"  # "AlexNet" and "GoogLeNet" and "ResNet" and "ResNet18"
    timestamp = int(time.time())  # 时间戳作为每次运行代码时的唯一标签

    # 参数设置
    max_epochs = 10
    lr = 0.003
    num_workers = 0
    lr_decay = 0.5  # 学习率衰减
    batch_size = 32
    weight_decay = 1e-3  # 权重衰减，L2 正则化，防止过拟合

    logdir = os.path.join(working_root, 'logs')  # 存放 tensorboard logs 文件
    # 没有则创建
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    log_file = os.path.join(working_root, 'logfile')  # 存放网络表现信息的 csv 文件
    # 没有则创建
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    load_param_dir = os.path.join(working_root, 'checkpoints')  # 存放训练好的参数
    # 没有则创建
    if not os.path.isdir(load_param_dir):
        os.makedirs(load_param_dir)

    # 加载参数，选择准确率最高的
    best_model_path = None
    _temp = [file.split('_')[0] for file in os.listdir(load_param_dir)]
    if model in _temp:
        param_logs = os.path.join(log_file, model + '_log.csv')
        _param_df = pd.read_csv(param_logs)
        best_model = _param_df.loc[_param_df["Validation Accuracy"].idxmax(), "Model File Name"]
        best_model_path = os.path.join(load_param_dir, best_model)

    print_feq = 20  # 打印信息频率

    label_dict = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        if config.real_or_try == "real":
            # 如果数据放在用户目录的 'AllData' 下则
            config.root = os.path.join(config.user_root, 'AllData', 'competitions', 'cifar-10')  # 本地设置: 数据目录
            config.res_path = os.path.join(config.working_root, 'sampleSubmission.csv')
        else:
            # 样本数据尝试
            config.root = os.path.join(config.working_root, 'TempData', 'competitions', 'cifar-10')
            config.res_path = os.path.join(config.working_root, 'result_example.csv')

        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('User config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


config = DefaultConfig()
