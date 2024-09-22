import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from env import *
from torchvision.transforms import ToTensor, ToPILImage

# 读取图像
image_path = r"E:\ANNprogram\data\BM\sample.png"  # 使用原始字符串
image = Image.open(image_path)

# 将图像转换为 Tensor
tensor_image = ToTensor()(image)

# 调用 show_img 函数显示原始图像
show_img(tensor_image)

# 将图像从 Tensor 转换回 NumPy 数组，并去掉 Alpha 通道
data = tensor_image.numpy().transpose(1, 2, 0)[:, :, :3]  # 去掉 Alpha 通道
print(data.shape)
#显示处理过的
plt.imshow(data)
plt.axis('off')  # 不显示坐标轴
plt.show()
# 定义血条的 RGB 范围
lower_bound = np.array([0.70, 0.70, 0.70])  # 示例低阈值
upper_bound = np.array([0.87, 0.87, 0.87])  # 示例高阈值

# 创建掩膜
mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=-1)

# 提取血条区域
blood_bar = np.zeros_like(data)
blood_bar[mask] = data[mask]

# 将血条区域转换为 Tensor
tensor_blood_bar = torch.from_numpy(blood_bar).permute(2, 0, 1)

# 显示血条区域
show_img(tensor_blood_bar)

# 提取非零像素
pixels = blood_bar[blood_bar > 0]
print(pixels)
print(pixels.size)