import numpy as np
import os
import sys
sys.path.append('..')
from utils.logger import Logger
from policies import POLICIES
from PIL import Image
import random
from mw_gym_make import mw_gym_make


os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

def visualize_images(imgs, env_name, task_id):
    key = f"Env name: {env_name}, task id: {task_id}"
    Logger.log({key: Logger.create_video(np.array(imgs), fps=40, format="mp4")})


def sample_demo_from_metaworld_env(env_name, policy, number_trajs, action_repeat, save_dir, resolution):
    env_save_dir = f'{save_dir}/{env_name}'
    for i in range(number_trajs):
        task_id = i % 50
        chosen_seed = 0
        env = mw_gym_make(env_name, seed=chosen_seed, task_id = task_id)[0]
        obs, info = env.reset()
        os.makedirs(env_save_dir, exist_ok=True)
        num_steps = (env.max_path_length) // action_repeat
        imgs = []
        states = []
        actions = []

        for step in range(num_steps):

            # TODO: figure out how to fix resolution
            if resolution:
                rgb_obs = env.render()
            else:
                rgb_obs = env.render()

            rgb_obs = np.array(Image.fromarray(rgb_obs).transpose(method=Image.FLIP_TOP_BOTTOM))
            imgs.append(rgb_obs.transpose(2, 0, 1))
            states.append(obs)
            action = policy.get_action(obs)
            actions.append(action)
            for _ in range(action_repeat):
                obs, reward, terminal, truncated,  info = env.step(action)
                if info['success'] == 1.0 or terminal or truncated:
                    break

            if info['success'] == 1.0:

                if resolution:
                    rgb_obs = env.render()
                else:
                    rgb_obs = env.render()

                rgb_obs = np.array(Image.fromarray(rgb_obs).transpose(method=Image.FLIP_TOP_BOTTOM))

                imgs.append(rgb_obs.transpose(2, 0, 1))
                states.append(obs)

                imgs = np.array(imgs)
                states = np.array(states)
                actions = np.array(actions)
                np.savez(f'{env_save_dir}/demontration_{i}_taskid_{task_id}_seed_{chosen_seed}.npz', imgs=imgs, states=states,
                        actions=actions)
                visualize_images(imgs = imgs, env_name = env_name, task_id=task_id)

                break

            elif terminal or truncated:
                print(f'the rollout failed: terminated {terminal} truncated {truncated} {env_name} {i}')
                break

        else: #
            print("exceeded the maximum allowable number of steps")


# Initialize wandb so we can visualize the demonstrations
Logger.init(project="metaworld_debug", config={}, dir="../scratch/wandb",
          name="demonstration_videos", entity="parameterized-skills-2")


save_dir = './mw_2500'
if os.path.exists(save_dir):
    raise Exception(f"save_dir {save_dir} already exists. Delete it or change the save dir")
    #shutil.rmtree(save_dir)

for env_name, policy in POLICIES.items():
    # env_name, policy, number_trajs, action_repeat, save_dir, resolution
    sample_demo_from_metaworld_env(env_name = env_name,
                                   policy = policy(),
                                   number_trajs=50,
                                   action_repeat=1,
                                   save_dir=save_dir,
                                   resolution = None)
