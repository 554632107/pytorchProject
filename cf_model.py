"""
    @time:2024/10/10 13:37
    @file:cf_model.py
"""
from torch import nn
import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

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


if __name__ == '__main__':
    test_cf = TestCF()
    input = torch.ones((64, 3, 32, 32))
    output = test_cf(input)
    print(output.shape)

