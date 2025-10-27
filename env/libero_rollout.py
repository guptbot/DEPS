import gc
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
from libero.lifelong.utils import *

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBS_MODALITIES = [('rgb', ['agentview_rgb', 'eye_in_hand_rgb']), ('depth', []), ('low_dim', ['gripper_states', 'joint_states'])]
OBS_KEY_MAPPING = {'agentview_rgb': 'agentview_image', 'eye_in_hand_rgb': 'robot0_eye_in_hand_image', 'gripper_states': 'robot0_gripper_qpos', 'joint_states': 'robot0_joint_pos'}

# Convert raw environment observations to tensor format for network input
def raw_obs_to_tensor_obs(obs, task_emb):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in OBS_MODALITIES:
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][OBS_KEY_MAPPING[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key]).unsqueeze(0)

    data = TensorUtils.map_tensor(data, lambda x: x.to(device))
    return data

# Wrapper for LIBERO environment to execute policy rollouts
class LiberoEnv():
    def __init__(self, traj_index, task, task_emb, args):
        traj_index = traj_index.item()
        task_emb = torch.flatten(task_emb).cpu()
        env_args = {
            "bddl_file_name": os.path.join(
                args.libero_root_path + "libero/libero/./bddl_files", task.problem_folder[0], task.bddl_file[0]
            ),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env_creation = False
        count = 0
        while not env_creation and count < 5:
            try:
                env = DummyVectorEnv(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(1)]
                )
                env_creation = True
            except Exception as e:
                print(f"Got exception {e} from LIBERO, retrying")
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")
        init_states_path = os.path.join(
            args.libero_root_path + "libero/libero/./init_files", task.problem_folder[0], task.init_states_file[0]
        )
        init_states = torch.load(init_states_path)
        self.env = env
        self.task_emb = task_emb
        self.env.reset()
        # Set initial state from saved trajectory
        init_states = np.expand_dims(init_states[traj_index], 0)
        obs = env.set_init_state(init_states)
        # Run dummy actions for physics stabilization
        dummy = np.zeros((1, 7))
        for _ in range(5):
            obs, _, _, _ = self.env.step(dummy)
        data = raw_obs_to_tensor_obs(obs, self.task_emb)
        self.data = data

        return None

    def get_data(self):
        return self.data

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.data = raw_obs_to_tensor_obs(obs, self.task_emb)
        return self.data, reward, done, info

    def terminate(self):
        self.env.close()
        gc.collect()



