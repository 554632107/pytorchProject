"""
    @time:2024/10/10 13:01
    @file:15_load_model.py
"""

import torch
import torchvision.models
from torch import nn

# 保存方式1加载模型
model1 = torch.load("vgg16_method1.pth")
# print(model1)

# 保存方式2加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
# print(vgg16)

# 不在一个py文件中加载自创的模型，要先进行类的导入，否则会报错

class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = torch.load("test_nn1.pth")
print(model)