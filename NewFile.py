import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pyautogui
import matplotlib.pyplot as plt

# 定义转换函数，将PIL图像转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 从屏幕输出获取224*224*3的Tensor
def get_img():
    img = pyautogui.screenshot(region=(0, 0, 224, 224))
    # 将PIL图像转换为Tensor
    img_tensor = transform(img)
    return img_tensor

# 显示图像
def show_img(img):
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

# 加载预训练模型
model_path = "E:/ANNprogram/model/resnet50-0676ba61.pth"
model = torchvision.models.resnet50(pretrained=False)  # 创建模型实例
model.load_state_dict(torch.load(model_path))  # 加载模型权重
model.eval()  # 设置模型为评估模式

# 打印模型大小信息通常是查看模型参数的数量
print(sum(p.numel() for p in model.parameters()))