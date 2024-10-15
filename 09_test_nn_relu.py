
import torch
from torch import nn
from torch.nn import MaxPool2d, ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

"""
    Relu 激活函数，引入非线性特征
"""



class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        # self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        self.relu1 = ReLU()
        # self.sigmod1 = Sigmoid()



    def forward(self, nn_input):
        nn_output  = self.relu1(nn_input)
        # sig_output = self.sigmod1(nn_input)
        return  nn_output




# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# print(output)
writer = SummaryWriter("CIFA10")

test_nn_relu = TestNN()

step = 0
for data in test_loader:
    imgs, labels = data
    writer.add_images("relu-input", imgs, step)
    nn_output = test_nn_relu(imgs)
    writer.add_images("relu-output", nn_output, step)
    step = step + 1

writer.close()

