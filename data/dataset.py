import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import config


class CIFAR10Dataset(Dataset):
    """加载数据集"""

    def __init__(self, root, trans=None, mode=None, label_dict=config.label_dict):
        """
        初始化
        :param root: 数据集文件路径
        :param trans: 变换操作
        :param mode: ['train', 'val', 'test']
        """
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.root = root

        if not label_dict:
            raise ValueError('label_dict should not be None')

        self.label_dict = label_dict  # 根据字典查询类别 id
        # 训练集的图片 id 与 label 对应表
        train_label_dir = os.path.join(self.root, 'trainLabels.csv')
        self.label_df = pd.read_csv(train_label_dir)

        # 训练/验证集
        train_val_dir = [os.path.join(self.root, 'inputs', 'train', i) for i in
                         os.listdir(os.path.join(self.root, 'inputs', 'train'))]

        if self.mode == 'test':
            # 测试
            self.img_dir = [os.path.join(self.root, 'inputs', 'test', i) for i in
                            os.listdir(os.path.join(self.root, 'inputs', 'test'))]
        elif self.mode == 'train':
            # 训练 70%
            self.img_dir = train_val_dir[:int(0.7 * len(train_val_dir))]
        else:
            # 验证 30%
            self.img_dir = train_val_dir[int(0.7 * len(train_val_dir)):]

        if trans is None:
            # 数据转换操作
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )

            self.trans = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        id_img = int(os.path.basename(img_path).split('.')[0])
        if self.mode == 'test':
            label = id_img
        else:
            label = self.label_dict[self.label_df[self.label_df['id'] == id_img]['label'].item()]

        data = Image.open(img_path)
        data = self.trans(data)

        return data, label

    def __len__(self):
        return len(self.img_dir)


if __name__ == '__main__':
    train_dataset = CIFAR10Dataset(root=config.root, mode='train', label_dict=config.label_dict)
    print(len(train_dataset))
    data, label = train_dataset[0]
    print(data.shape)
    print(label)

    val_dataset = CIFAR10Dataset(root=config.root, mode='val', label_dict=config.label_dict)
    print(len(val_dataset))
    data, label = val_dataset[0]
    print(data.shape)
    print(label)

    test_dataset = CIFAR10Dataset(root=config.root, mode='test', label_dict=config.label_dict)
    print(len(test_dataset))
    data, label = test_dataset[0]
    print(data.shape)
    print(label)
