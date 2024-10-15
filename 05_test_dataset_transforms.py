
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data/CIFAdata", train=False,  transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)

writer = SummaryWriter("CIFA10")
for i in range(10):
    img, label= test_set[i]
    print(label)
    writer.add_image("test_set", img, i)

writer.close()
# img, label = test_set[0]
# img.show()