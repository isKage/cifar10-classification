import time

import pandas as pd

import models
from config import config
from data import CIFAR10Dataset

import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import py7zr


def unzip(**kwargs):
    config._parse(kwargs)
    
    """解压数据集 【执行一次即可!!!】"""
    extract_dir = os.path.join(config.root, 'inputs')
    os.makedirs(extract_dir, exist_ok=True)
    print('Extracting files...', extract_dir)

    # 解压训练集，执行一次即可
    with py7zr.SevenZipFile(os.path.join(config.root, 'train.7z'), mode='r') as archive:
        archive.extractall(path=extract_dir)

    with py7zr.SevenZipFile(os.path.join(config.root, 'test.7z'), mode='r') as archive:
        archive.extractall(path=extract_dir)


def train(**kwargs):
    config._parse(kwargs)

    classification = getattr(models, config.model)()  # 网络模型
    classification.to(config.device)

    train_dataset = CIFAR10Dataset(root=config.root, mode="train")  # 训练集
    val_dataset = CIFAR10Dataset(root=config.root, mode="val")  # 验证集

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
        )
    val_data_size = len(val_dataset)  # 验证集长度

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()  # 分类常用交叉熵

    # 优化器
    lr = config.lr
    optimizer = torch.optim.SGD(
        params=classification.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,  # L2 正则化 权重衰减
    )

    # 设置训练网络的参数
    epochs = config.max_epochs  # 训练迭代次数，即训练几轮

    # 添加tensorboard可视化
    writer = SummaryWriter(config.logdir)

    # 打开日志文件（如果没有则创建, 如果有则继续追加）
    log_file_path = os.path.join(config.log_file, '{}_log.csv'.format(config.model))
    log_file_exists = os.path.exists(log_file_path)
    log_file = open(log_file_path, 'a')
    if not log_file_exists:
        log_file.write(
            "TimeStamp,Epoch,Train Loss,Validation Loss,Validation Accuracy,Learning Rate,Model File Name\n"
        )  # 写入表头

    # 开始训练
    previous_loss = 1e10  # 初始化误差
    for epoch in range(epochs):
        print("------------- 第 {} 轮训练开始 -------------".format(epoch + 1))

        # 训练步骤
        classification.train()

        total_train_step = 0  # 训练次数
        train_ave_loss = 0.0  # 训练平均 loss
        for data in train_dataloader:
            # 输入输出
            images, targets = data
            images, targets = images.to(config.device), targets.to(config.device)

            # 前向传播
            outputs = classification(images)

            # 损失函数
            loss = loss_fn(outputs, targets)
            train_ave_loss += loss.item()

            # 清零梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            total_train_step += 1
            if total_train_step % config.print_feq == 0:
                print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar(
                    tag="Loss in Train Set (every {} steps) in Epoch {}".format(config.print_feq, epoch + 1),
                    scalar_value=loss.item(),
                    global_step=total_train_step,
                )
        train_ave_loss /= total_train_step

        # 测试步骤 (不更新参数)
        classification.eval()
        total_val_loss = 0  # 测试集损失累积
        total_accuracy = 0  # 分类问题正确率
        with torch.no_grad():
            for data in val_dataloader:
                images, targets = data
                images, targets = images.to(config.device), targets.to(config.device)

                outputs = classification(images)

                loss = loss_fn(outputs, targets)

                # 损失
                total_val_loss += loss.item()

                # 正确率
                accuracy = (outputs.argmax(axis=1) == targets).sum()
                total_accuracy += accuracy.item()

        # 可视化: 在验证集上的损失
        print("##### 第 {} 轮: 在测试集上的 loss: {} #####".format(epoch + 1, total_val_loss))
        writer.add_scalar(
            tag="Loss in Validation Set",
            scalar_value=total_val_loss,
            global_step=epoch,
        )

        # 可视化: 在验证集上的正确率
        print("##### 第 {} 轮: 在测试集上的正确率: {} #####".format(epoch + 1, total_accuracy / val_data_size))
        writer.add_scalar(
            tag="Accuracy in Validation Set",
            scalar_value=total_accuracy / val_data_size,
            global_step=epoch,
        )

        # 保存每次训练的模型
        file_name = classification.save()  # 保存
        print("##### 模型成功保存在 {} #####".format(file_name))

        # 更新学习率
        if total_val_loss / val_data_size > previous_loss:
            lr = lr * config.lr_decay
            print("更新学习率为 {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = total_val_loss / val_data_size

        # 写入日志文件 logs.txt
        log_file.write(
            "{time},{epoch},{train_loss:.4f},{val_loss:.4f},{accuracy:.4f},{lr:.6f},{file_name}\n".format(
                time=config.timestamp,  # 时间戳作为每次运行代码时的唯一标签
                epoch=epoch + 1,
                train_loss=train_ave_loss,
                val_loss=total_val_loss / val_data_size,
                accuracy=total_accuracy / val_data_size,
                lr=lr,
                file_name=file_name
            )
        )

    log_file.close()  # 关闭日志文件
    writer.close()  # 关闭 tensorboard 文件


@torch.no_grad()
def test(**kwargs):
    config._parse(kwargs)

    # configure model
    model = getattr(models, config.model)().eval()
    if config.best_model_path:
        model.load(config.best_model_path)
    else:
        raise ValueError("No model! Please train the model first!")

    model.to(config.device)

    # data
    test_dataset = CIFAR10Dataset(config.root, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)

    test_dataloader_size = len(test_dataloader)

    # results = []

    # 测试集结果 csv 表格
    res_path = config.res_path
    res_path_exists = os.path.exists(res_path)
    res = open(res_path, 'a')
    # 如果存在则不写表头，紧接着填充数据。只有不存在时 not res_path_exists == True 才写入表头
    if not res_path_exists:
        res.write(
            "id,label\n"
        )  # 写入表头

    batch = 0
    for data, path in test_dataloader:
        start = time.time()
        batch += 1

        inputs = data.to(config.device)
        score = model(inputs)

        predicted_label_num = score.max(dim=1)[1].detach().tolist()

        # 将数字标签转换为对应的名称
        predicted_label = []
        for num_label in predicted_label_num:
            for key, val in config.label_dict.items():
                if num_label == val:
                    predicted_label.append(key)
        for path_, label_ in zip(path, predicted_label):
            res.write("{id},{label}\n".format(id=path_.item(), label=label_))

        end = time.time()

        print("A Batch Done: {} / {}, cost of time {}".format(batch, test_dataloader_size, end - start))

        # 保存为 id, label 的格式 (全部读取再写入，太消耗内存)
        # batch_results = [(path_.item(), label_) for path_, label_ in zip(path, predicted_label)]
        # results += batch_results

    res.close()

    # write_csv(results, config.result_file)
    print("Test Result saved in\n {}".format(res_path))

    # return results[:5]


def write_csv(results, file_name):
    import csv
    # 按照 id 从小到大排序
    sorted_results = sorted(results, key=lambda x: x[0])

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(sorted_results)


def sort_csv(**kwargs):
    config._parse(kwargs)

    # 排序 id
    df = pd.read_csv(config.res_path)
    df = df.sort_values(by='id')
    df.to_csv(config.res_path, index=False)
    print("Sorting Result saved in\n {}".format(config.res_path))


def help():
    """打印帮助的信息： python main.py help"""
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --model=ResNet --lr=0.01
            python {0} test --load=./loads_param_dir/ResNet.pth
            python {0} help
    """.format(__file__))


if __name__ == '__main__':
    import fire

    fire.Fire()
