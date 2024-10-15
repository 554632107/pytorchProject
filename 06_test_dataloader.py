"""
   dataloader 负责将dataset数据集中的数据导入到神经网络
   参数：dataset, batch_size（每次取多少samples）, shuffle(是否打乱数据)
   num_workers(多进程),sampler默认是随机抓取,drop_last=True当取的最后一批次数据的数量不足时舍弃
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

img, label = test_data[0]
print(img.shape)
print(label)
writer = SummaryWriter("dataloader")

# epoch 取数据的轮次
for epoch in range(2):
    print(epoch)
    step = 0
    for data in test_loader:
        imgs, labels = data
        # print(imgs.shape)
        # print(labels)
        writer.add_images("Epoch:{}".format(epoch),imgs, step)
        step = step + 1


writer.close()