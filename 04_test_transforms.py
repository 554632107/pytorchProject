from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


"""
    torchvision.transforms模块提供了一系列用于图像预处理和数据增强的操作。这些操作可以方便地组合在一起，对图像数据进行转换，使其更适合深度学习模型的训练和推理。
    transforms中的常用功能：
    数据归一化（transforms.Normalize）：对图像的像素值进行归一化处理。通常是将每个通道（如 RGB 图像的红、绿、蓝通道）的像素值减去均值并除以标准差。这有助于模型更快地收敛和提高性能。
    调整大小（transforms.Resize）：改变图像的尺寸。这在将不同大小的图像输入到模型中时非常有用，因为模型通常需要固定大小的输入。
    中心裁剪（transforms.CenterCrop）：从图像的中心区域裁剪出指定大小的图像。这可以帮助聚焦图像的关键部分，去除边缘可能存在的无关信息。
    转换为张量（transforms.ToTensor）:将图像数据从其他格式（如numpy.ndarray）转换为PyTorch张量。同时，会将像素值的范围从[0, 255]转换为[0, 1]，并且将通道顺序从HWC（高度、宽度、通道）转换为CHW（通道、高度、宽度），这是PyTorch模型通常所要求的输入格式。
    组合变换（transforms.Compose）：主要功能是将多个图像变换（transform）操作组合成一个单一的变换操作序列。这在处理图像数据时非常有用，因为在深度学习中，通常需要对图像进行多个预处理步骤，如调整大小、裁剪、归一化、转换为张量等
"""
img_path = "data/dataAB/train/ants_image/0013035.jpg"
img_path_abs = "/data/dataAB/train/ants_image/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

print(img)
# ToTensor
trans_totensor = transforms.ToTensor() # 类的实例化
tensor_img = trans_totensor(img)
print(tensor_img)
writer.add_image("Tensor_img", tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.1, 1, 0.1], [1, 1, 1] )
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 3)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(30)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_corp = trans_compose_2(img)
    writer.add_image("RandomCrop", img_corp, i)

writer.close()
