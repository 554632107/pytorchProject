"""
    @time:2024/10/9 22:31
    @file:14_test_nn.py
"""

# vgg16 模型修改

import torchvision
from torch import nn
# 仅加载网络模型，不下载
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 下载网络模型
vgg16_true = torchvision.models.vgg16(pretrained=True)


# print(vgg16_false)
print(vgg16_true)
# 对classifier添加module
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 修改classifier(6)线性层参数
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)