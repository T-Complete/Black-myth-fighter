import torch
import time
import threading
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pyautogui
from PIL import Image
import matplotlib.pyplot as plt

# 定义转换函数，将PIL图像转换为3*224*224的Tensor
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# 从屏幕输出获取图片
def get_img():
    return  pyautogui.screenshot()

# 显示图像
def show_img(img):
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

class envs():
    def __init__(self,flash_time=0.1):
        self.action_space = [
    'a', 'd', 'w', 's', 'q', 'e', 'r', 'z', 'x', 'c', 'v', ' ',
    'ctrl', 'ml', 'mm', 'mr',
    'move_up', 'move_down', 'move_left', 'move_right',
    'none', 'none', 'none'
    ]
        self.observation_space = (224, 224, 3)
        self.action_dim = len(self.action_space)
        self.flash_time = flash_time
        ##上个时刻血量、蓝量、体力：
        self.health=1000
        self.mana=100
        self.stamina=100

        self.line_number=[] ##行号，待测定
        self.fcolour=[] ##颜色，待测定
        self.findready=[] ##是否可以点击法术，待测定

    def _simulate_key_press(self, key):
        if key == 'ctrl':
            pyautogui.keyDown('ctrl')
            time.sleep(self.flash_time)
            pyautogui.keyUp('ctrl')
        elif key in ['ml', 'mm', 'mr']:
            if key == 'ml':
                pyautogui.click(button='left')
            elif key == 'mm':
                pyautogui.click(button='middle')
            elif key == 'mr':
                pyautogui.click(button='right')
        elif key in ['move_up', 'move_down', 'move_left', 'move_right']:
            if key == 'move_up':
                pyautogui.moveRel(0, -10)
            elif key == 'move_down':
                pyautogui.moveRel(0, 10)
            elif key == 'move_left':
                pyautogui.moveRel(-10, 0)
            elif key == 'move_right':
                pyautogui.moveRel(10, 0)
        else:
            pyautogui.keyDown(key)
            time.sleep(self.flash_time)
            pyautogui.keyUp(key)
    from PIL import Image

def find_color_length(image_path, y, target_color):
    # 打开图片
    img = Image.open(image_path)
    
    # 获取图片尺寸
    width, height = img.size
    
    # 初始化连续颜色长度
    length = 0
    max_length = 0
    
    # 遍历指定行的所有像素
    for x in range(width):
        # 获取像素颜色
        pixel_color = img.getpixel((x, y))
        
        # 如果颜色匹配，则增加长度
        if pixel_color == target_color:
            length += 1
            max_length = max(max_length, length)
        else:
            # 如果颜色不匹配，则重置长度
            length = 0
    
    return max_length

    ##通过分析tensor对应图像指定区域像素获得当前任务状态
    def get_reward(self,img):
        ##识别图像进行状态识别，待完善
        return reward
    def step(self,actions_tensor):
        if not isinstance(actions_tensor,torch.Tensor):
            raise TypeError('actions_tensor must be a torch.Tensor')

        _,top_indices = torch.topk(actions_tensor, 2)
        # 将索引转换为对应的按键
        keys = [self.action_space[index.item()] for index in top_indices]

        # 创建线程列表
        threads = []

        # 启动线程
        for key in keys:
            thread = threading.Thread(target=self._simulate_key_press, args=(key,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()