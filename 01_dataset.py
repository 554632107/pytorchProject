import os
from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "/test/data/hymenoptera_data/train"
ants_dir = "ants"
bees_dir = "bees"
ants_dataset = MyData(root_dir, ants_dir)
print(len(ants_dataset))
bees_dataset = MyData(root_dir, bees_dir)
print(len(bees_dataset))
train_dataset = ants_dataset + bees_dataset
img, label = train_dataset[131]
# img.show()
print(img.filename)
print(label)