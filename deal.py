from os import wait
import numpy as np
import pyautogui
from PIL import Image
from env import *
import time
from torchvision.transforms import ToTensor

stop_event= threading.Event()
class Agent:
    def __init__(self, model,envs):
        self.model = model
        self.env = envs
        self.action_space = self.env.action_space
        pass # TODO more

def playing_access(env, Agent):
    experience_buffer = []  # 用于存储经验
    while not stop_event.is_set():
        start_time = time.time()
        
        # 获取当前状态
        state = env.obs()
        action = Agent.forward(state)  # 选择动作
        env.step(action)  # 执行动作
        
        # 获取奖励和下一个状态
        rewards = env.rewards()
        next_state = env.obs()  # 获取下一个状态
        done = env.is_done()  # 检查是否结束
        
        # 存储经验
        experience_buffer.append((state, action, rewards, next_state, done))
        
        # 训练模型
        if len(experience_buffer) >= Agent.batch_size:  # 如果经验缓冲区达到一定大小
            # 从经验中采样
            batch = np.random.choice(len(experience_buffer), Agent.batch_size, replace=False)
            experience = [experience_buffer[i] for i in batch]
            Agent.train(experience)  # 训练代理

        print(rewards)

        elapsed_time = time.time() - start_time
        wait_time = max(0, wait_time - elapsed_time)
        if wait_time != 0:
            time.sleep(wait_time)
