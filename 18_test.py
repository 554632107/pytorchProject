"""
    @time:2024/10/10 20:26
    @file:18_test.py
"""
import torchvision
from PIL import Image
from mpmath.identification import transforms
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


# 定义训练的设备
train_device = torch.device("mps")

image_path = "imgs/plane.png"
image = Image.open(image_path)
# png格式的图片时4个通道
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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

test_cf = torch.load("test_cf9.pth", weights_only=False)
print(test_cf)
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(train_device)
test_cf.eval()
with torch.no_grad():
    output = test_cf(image)
    output.to(train_device)
print(output)
print(output.argmax(1))