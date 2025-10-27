import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mw.metaworld_tasks import TASK_IDX_TO_TASK_NAME
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PreloadedNPZDataset(Dataset):
    def __init__(self, root_dir, task_indices, trajectories, max_traj_len):

        self.data = []
        self.max_traj_len = max_traj_len
        self.task_lengths = []

        for task_idx in task_indices:
            print(f"Loading task {task_idx}")
            path = os.path.join(root_dir, TASK_IDX_TO_TASK_NAME[task_idx])
            task_demonstration_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            assert 50 >= len(task_demonstration_files) >= len(trajectories) - 5
            for traj_idx in range(min(len(task_demonstration_files), len(trajectories))):
                file_path = task_demonstration_files[traj_idx]
                file_path = os.path.join(path, file_path)
                self.data.append(self.retrieve_demonstration(file_path, task_idx))
            print(f"Loaded {traj_idx + 1} trajectories")

        #print("task lengths", self.task_lengths)
        print(f"min length {min(self.task_lengths)}")
        print(f"max length {max(self.task_lengths)}")
        #print(self.task_lengths)


    def retrieve_demonstration(self, demonstration_path, task_idx):
        data = np.load(demonstration_path)
        imgs = torch.from_numpy(data['imgs'])[:-1]
        robot_states = torch.from_numpy(self.retrieve_robot_state(data['states']))[:-1]
        actions = np.clip(data['actions'], -1, 1)
        actions = torch.from_numpy(actions)

        task_length = imgs.shape[0]
        self.task_lengths.append(task_length)

        return {
            "imgs": self.pad_demonstration(imgs).float(),
            "robot_states": self.pad_demonstration(robot_states).float(),
            "actions": self.pad_demonstration(actions).float(),
            "task_index": torch.tensor(task_idx),
            "task_length": torch.tensor(task_length)
        }

    def pad_demonstration(self, demonstration):
        rows = demonstration.shape[0]
        extra_rows = self.max_traj_len - rows
        assert extra_rows >= 0
        zeros = torch.zeros((extra_rows,) +  demonstration.shape[1:], dtype=demonstration.dtype)
        return torch.cat((demonstration, zeros), dim = 0)

    def retrieve_robot_state(self, states):
        return np.hstack((states[:, :4], states[:, 18 : 18 + 4]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample


def fetch_dataloader(datasets_path, batch_size, tasks, trajectories, max_traj_len):
    dataset = PreloadedNPZDataset(datasets_path, tasks, trajectories, max_traj_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=RandomSampler(dataset),
        persistent_workers=False)

    return dataloader


if __name__ == "__main__":
    dataloader = fetch_dataloader("./mw/metaworld_expert_demonstrations_ds", 4, list(range(50)), list(range(30)), 500)
