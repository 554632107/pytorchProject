
import torch
from torch.nn import Conv2d
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x

test_nn = TestNN()
print(test_nn)

writer = SummaryWriter("CIFA10")
step = 0
for data in test_loader:
    imgs, labels = data
    output = test_nn(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1


writer.close()