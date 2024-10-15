import torch
import torchvision.datasets
from torch.utils.data import DataLoader

from requests.packages import target
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=False, num_workers=0, drop_last=True)


class TestNnSeq(nn.Module):
    def __init__(self):
        super(TestNnSeq, self).__init__()
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
        x= self.modle1(x)
        return x

loss = nn.CrossEntropyLoss()
test_seq = TestNnSeq()
optim = torch.optim.SGD(test_seq.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in data_loader:
        imgs, targets = data
        outputs = test_seq(imgs)
        # print("out:", outputs)
        # print("tar:", targets)
        result_loss = loss(outputs, targets)
        # print(result_loss)
        optim.zero_grad() # 梯度清0
        result_loss.backward() # 反向传播
        optim.step()  # 优化参数
        running_loss = running_loss + result_loss

    print(running_loss)

# writer = SummaryWriter("logs_seq")
# writer.add_graph(test_seq, test_input)
# writer.close()





