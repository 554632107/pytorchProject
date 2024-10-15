
import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

"""
    最大池化
"""



input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class TestNnMaxpool(nn.Module):
    def __init__(self):
        super(TestNnMaxpool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)


    def forward(self, input):
        output  = self.maxpool1(input)
        return output


test_nn_maxpool = TestNnMaxpool()
output = test_nn_maxpool(nn_input)
print(output)

# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# print(output)
writer = SummaryWriter("CIFA10")
step = 0
for data in test_loader:
    imgs, labels = data
    writer.add_images("maxpool-input", imgs, step)
    output = test_nn_maxpool(imgs)
    writer.add_images("maxpool-output", output, step)
    step = step + 1

writer.close()

