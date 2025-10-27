import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from metaworld_tasks import TASK_IDX_TO_TASK_NAME
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import sys
sys.path.append('..')
from utils.logger import Logger


def downsample_demonstrations(root_dir, new_dir):

    for task_idx in range(50):
        print(f"Loading task {task_idx}")
        path = os.path.join(root_dir, TASK_IDX_TO_TASK_NAME[task_idx])
        new_path = os.path.join(new_dir, TASK_IDX_TO_TASK_NAME[task_idx])
        os.makedirs(new_path, exist_ok=True)
        task_demonstration_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for idx, file_path in enumerate(task_demonstration_files):
            print("loading demo")
            old_file_path = os.path.join(path, file_path)
            new_file_path = os.path.join(new_path, file_path)
            key = f"Task: {TASK_IDX_TO_TASK_NAME[task_idx]} trajectory: {idx}"
            process_single_trajctory(old_file_path, new_file_path, key)


def visualize_images(imgs, key):
    Logger.log({key: Logger.create_video(np.array(imgs), fps=40, format="mp4")})


def process_single_trajctory(path, new_path, key):
    data = np.load(path)
    imgs = data['imgs']
    resized = resize_images(imgs)

    np.savez(new_path, imgs=resized, states=data['states'],
             actions=data['actions'])
    visualize_images(resized, key)
    print("processed trajectory")

def resize_images(imgs):
    b, _, _, _ = imgs.shape
    resized = np.zeros((b, 3, 84, 84), dtype = imgs.dtype)
    for x in range(b):
        img = np.transpose(imgs[x], (1, 2, 0))
        new_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        resized[x] = np.transpose(new_img, (2, 0, 1))
    return resized


Logger.init(project="metaworld_debug", config={}, dir="../scratch/wandb",
          name="demonstration_videos_downsampled", entity="parameterized-skills-2")

downsample_demonstrations("./mw_2500", "./mw_2500_ds")

