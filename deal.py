import os
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
    experience_buffer = []  # ���ڴ洢����
    while not stop_event.is_set():
        start_time = time.time()
        
        # ��ȡ��ǰ״̬
        state = env.obs()
        action = Agent.forward(state)  # ѡ����
        env.step(action)  # ִ�ж���
        
        # ��ȡ��������һ��״̬
        rewards = env.rewards()
        next_state = env.obs()  # ��ȡ��һ��״̬
        done = env.is_done()  # ����Ƿ����
        
        # �洢����
        experience_buffer.append((state, action, rewards, next_state, done))
        
        # ѵ��ģ��
        if len(experience_buffer) >= Agent.batch_size:  # ������黺�����ﵽһ����С
            # �Ӿ����в���
            batch = np.random.choice(len(experience_buffer), Agent.batch_size, replace=False)
            experience = [experience_buffer[i] for i in batch]
            Agent.train(experience)  # ѵ������

        print(rewards)

        elapsed_time = time.time() - start_time
        wait_time = max(0, wait_time - elapsed_time)
        if wait_time != 0:
            time.sleep(wait_time)

