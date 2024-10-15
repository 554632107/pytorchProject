
import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

"""
    线性层
"""



class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)




    def forward(self, nn_input):
        nn_output  = self.linear1(nn_input)
        return  nn_output




# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# print(output)
writer = SummaryWriter("CIFA10")

test_nn_linear = TestNN()

step = 0
for data in test_loader:
    imgs, labels = data
    writer.add_images("linear-input", imgs, step)
    nn_input = torch.reshape(imgs, (1, 1, 1, -1))
    # nn_input = torch.flatten(imgs)  # 展平
    print("nn_input", nn_input)
    nn_output = test_nn_linear(nn_input)
    print("nn_output", nn_output)
    writer.add_images("linear-output", nn_output, step)
    step = step + 1

writer.close()

