"""
    @time:2024/10/10 12:57
    @file:15_save_model.py
    模型的保存/模型的加载
"""
from torchvision.models import vgg16
import torch
from torch import nn

vgg16 = vgg16(pretrained=False)

# 保存方式1 保存了网络模型的结构和参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 保存网络模型的参数,字典（官方推荐，只能用空间小）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

test_nn = TestNN()
torch.save(test_nn, "test_nn1.pth")
