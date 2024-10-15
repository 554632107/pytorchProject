"""
    @time:2024/10/10 13:27
    @file:16_train.py
    完整的模型训练套路
"""
import torch.optim
from requests.packages import target
from torch.utils.data import DataLoader
from cf_model import *
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

dataset_transform = torchvision.transforms.ToTensor()
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=True, transform=dataset_transform, download=True)
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=dataset_transform, download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度{}".format(train_data_size))
print("测试数据集的长度{}".format(test_data_size))

# 利用DataLoader来加载数据集

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(train_data, batch_size=64)

# 搭建神经网络
test_cf = TestCF()

# 损失函数
loss_cf = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01 # 1e-2 = 1*(10)^(-2)=1/100=0.01
optimizer = torch.optim.SGD(test_cf.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("log_train")

for i in range(epoch):
    print("-------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    test_cf.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = test_cf(imgs)
        loss = loss_cf(outputs, targets)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    test_cf.eval()
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = test_cf(imgs)
            loss = loss_cf(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(test_cf, "test_cf{}.pth".format(i))
    print("模型已保存")
writer.close()
