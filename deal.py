import numpy as np
import pyautogui
from PIL import Image
from env import *
from torchvision.transforms import ToTensor

class Agent:
    def __init__(self, model,envs):
        self.model = model
        self.env = envs
        self.action_space = self.env.action_space
        pass # TODO more
    def runit():
        pass   #TODO more


img=Image.open("E:\ANNprogram\data\BM\sample.png") #path/to/your/image
plt.imshow(img)
plt.axis('off')
plt.show()