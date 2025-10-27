import gc
import numpy as np

import torch
import random
from mw.mw_gym_make import mw_gym_make
import cv2
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

class MetaworldRollout():
    def __init__(self, task, traj_index):
        chosen_seed = 0 #random.choice(range(int(1e5)))
        env = mw_gym_make(task, seed=chosen_seed, task_id=traj_index)[0]
        state, info = env.reset()
        img = env.render()

        self.env = env
        self.obs = self.downsample_image(img)
        self.robot_state = self.retrieve_robot_state(state)

    def downsample_image(self, img):
        assert len(img.shape) == 3
        img = np.array(Image.fromarray(img).transpose(method=Image.FLIP_TOP_BOTTOM))
        new_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        return np.transpose(new_img, (2, 0, 1))

    def retrieve_robot_state(self, state):
        return np.hstack((state[:4], state[18 : 18 + 4]))

    def get_data(self):
        return {"imgs": torch.from_numpy(self.obs).float().reshape(1, 1, 3, 84, 84).to(device),
                "robot_states": torch.from_numpy(self.robot_state).float().reshape(1, 1, -1).to(device)}

    def step(self, action):
        state, reward, terminal, truncated, info = self.env.step(action)
        img = self.env.render()
        self.obs = self.downsample_image(img)
        self.robot_state = self.retrieve_robot_state(state)

        obs_dict = self.get_data()

        return obs_dict, reward, terminal, truncated, info

    def terminate(self):
        self.env.close()
        gc.collect()
        pass



