"""
    @time:2024/10/10 19:42
    @file:train_gpu1.py
"""
import torch.optim
from requests.packages import target
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.xpu import device
from urllib3.filepost import writer
import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import time

# 定义训练的设备
train_device = torch.device("mps")

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

class TestCF(nn.Module):
    def __init__(self):
        super(TestCF, self).__init__()
        self.modle1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)

        )

    def forward(self, x):
        x = self.modle1(x)
        return x

test_cf = TestCF()
test_cf.to(train_device)

# 损失函数
loss_cf = nn.CrossEntropyLoss()
loss_cf = loss_cf.to(train_device)
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
start_time = time.time()
for i in range(epoch):
    print("-------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    test_cf.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(train_device)
        targets = targets.to(train_device)
        outputs = test_cf(imgs)
        loss = loss_cf(outputs, targets)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练时间", end_time - start_time)
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    test_cf.eval()
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs = imgs.to(train_device)
            targets = targets.to(train_device)
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
