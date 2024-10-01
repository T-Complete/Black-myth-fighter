import torch
import time
import threading
import torch.nn.functional as F
import torchvision
import concurrent.futures
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

# 指定区域的像素统计
def count_pixels_in_range(region, value_range):
    # 获取屏幕截图
    screenshot = pyautogui.screenshot(region=region)
    
    # 删除alpha通道，转换为RGB格式
    img_rgb = screenshot.convert('RGB')
    
    # 转换为Tensor即NumPy数组
    img_array = torch.tensor(list(img_rgb.getdata()), dtype=torch.float32).view(img_rgb.size[1], img_rgb.size[0], 3)

    # 定义要统计的范围
    lower_bound = torch.tensor(value_range[0], dtype=torch.float32)
    upper_bound = torch.tensor(value_range[1], dtype=torch.float32)

    # 计算在范围内的像素
    mask = (img_array >= lower_bound) & (img_array <= upper_bound)
    count = mask.all(dim=-1).sum().item()  # 在三个通道都满足条件时计数

    return count

#  ===========example========== #
## 定义截图区域 (left, top, width, height)
#region = (100, 100, 300, 300)  # 示例区域
## 定义要统计的像素值范围 (lower_bound, upper_bound)
#value_range = ((50, 50, 50), (200, 200, 200))  # 示例范围，注意这里不再有alpha通道


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
    'none', 'none'
    ]
        self.observation_space = (224, 224, 3)
        self.action_dim = len(self.action_space)
        self.flash_time = flash_time
        self.reward_weight = torch.tensor([5,3,0.2])
        self.region = [(150,1360,556,1461),
                       (150,1360,556,1461),
                       (150,1360,556,1461),#状态条范围
                       ()#boss血条，待测
                       ()#小怪血条，待测
            ]
        self.colours = [
            ((200,200,200)(256,256,256)),#白色，血量
            ((60,70,160),(70,80,180)),#蓝量
            ((185,150,95),(195,160,105))#体力
            ((200,200,200)(256,256,256)),#boss,白色
            ((),())#小怪，待测定
            ]
        with concurrent.futures.ThreadPoolExecutor() as executor:##创建线程池
            futures=[executor.submit(count_pixels_in_range, self.region[i],self.colours[i]) for i in range[0,3]]##提交任务获取返回值
            results = [future.result() for future in futures]
            last_list = torch.tensor(results)


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
    
    def obs():
        img=get_img()
        img=transform_image(img)
        return img
    ##通过分析tensor对应图像指定区域像素获得当前任务状态
    def reward(self,img):
        threads = []
        reward = -10;
        with concurrent.futures.ThreadPoolExecutor() as executor:##创建线程池
            futures=[executor.submit(count_pixels_in_range, self.region[i],self.colours[i]) for i in range[0,3]]##提交任务获取返回值
            results = [future.result() for future in futures]
            tensor_result = torch.tensor(results)
            reward = torch.sum((tensor_result-self.last_list)*self.reward_weight) ##计算奖励
            last_list = tensor_result
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