from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image



"""
启动 tensorboard
tensorboard --logdir=logs
tensorboard --logdir=logs --port=6007
"""
writer = SummaryWriter("logs")


"""
    将PIL读取的图片转为numpy数据类型
"""
image_path = "/data/dataAB/train/bees_image/16838648_415acd9e3f.jpg"
img = Image.open(image_path)

print(type(img))
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)
# (512, 768, 3) 通道数为3，数据格式通道数在后面，所以要指定dataformats="HWC"
writer.add_image("test", img_array, 2, dataformats="HWC")
# add_image中默认dataformats="CHW" 通道数C、高度H、宽度w，从PIL到numpy,需要在add_image()中指定shape中每一个数字/维表示的含义


"""
  利用opencv读取图片，获得numpy型图片数据
"""

# add_scalar的用法，y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i , i)

writer.close()


