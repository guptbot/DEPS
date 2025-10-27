import os
import h5py
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader

LANG_EMBEDDING_CFG = EasyDict({'task_embedding_format':'bert', 'data':{'max_word_len':25}, 'policy':{'language_encoder':{'network_kwargs':{'input_size':0}}}})

# Wrapper dataset that pads trajectories to fixed length and zeros out padding
class ZeroPaddedDataset(Dataset):
    def __init__(self, sequence_dataset, task_lengths, task_info, task_idx):
        self.sequence_dataset = sequence_dataset
        self.task_lengths = task_lengths
        self.n_demos = len(task_lengths)
        self.total_num_sequences = len(task_lengths)
        self.task_info = task_info
        self.task_idx = task_idx

    def __len__(self):
        return len(self.sequence_dataset)
    def change_padding_to_zero(self, traj, task_length):
        if isinstance(traj, dict):
            for key in traj.keys():
                self.change_padding_to_zero(traj[key], task_length)
        elif isinstance(traj, np.ndarray):
            traj[task_length:] = 0
        else:
            raise Exception("Something went wrong!!!")
    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        self.change_padding_to_zero(return_dict, self.task_lengths[idx])
        return_dict['task_length'] = self.task_lengths[idx]
        return_dict["task_info"] = self.task_info
        return_dict["task_index"] = self.task_idx
        return return_dict

# Filter dataset to specific trajectories and apply zero padding
def clean_dataset(dataset, traj_lengths, task_info, idx, trajectories):
    '''
    1 - Remove all elements that do not correspond to a new trajectory
    2 - Replace end padding with zeroes
    '''
    indices_to_keep = [sum(traj_lengths[0:i]) for i in range (len(traj_lengths) + 1)]
    indices_to_keep = indices_to_keep[:-1]
    assert len(indices_to_keep) == 50
    # Filter out the elements not in `trajectories`
    indices_to_keep = [indices_to_keep[i] for i in trajectories]
    traj_lengths = [traj_lengths[i] for i in trajectories]
    dataset = Subset(dataset, indices_to_keep)
    return ZeroPaddedDataset(dataset, traj_lengths, task_info, idx)

def get_dataset_info(dataset_path, filter_key=None, verbose=True):
    # extract demonstration list from file
    all_filter_keys = None
    f = h5py.File(dataset_path, "r")
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted(
            [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        )
    else:
        # use all demonstrations
        demos = sorted(list(f["data"].keys()))

        # extract filter key information
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                )
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)
    return traj_lengths

# Create dataloader for LIBERO benchmark tasks
def fetch_dataloader(datasets_path, benchmark_name, batch_size, tasks, trajectories, max_traj_len):
    benchmark = get_benchmark(benchmark_name)()
    n_manip_tasks = benchmark.n_tasks
    obs_modality = EasyDict({'rgb': ['agentview_rgb', 'eye_in_hand_rgb'], 'depth': [], 'low_dim': ['gripper_states', 'joint_states']})
    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    traj_lengths = [None for _ in range(benchmark.n_tasks)]
    task_info = [None for _ in range(benchmark.n_tasks)]
    for i in tasks: # currently we assume tasks from same benchmark have the same shape_meta
        try:
            dataset_path = os.path.join(datasets_path, benchmark.get_task_demonstration(i))
            traj_lengths[i] = get_dataset_info(dataset_path)
            task_info[i] = benchmark.get_task(i)
        except Exception as e:
            print(f"{e}: [error] failed to load task {i} name {benchmark.get_task_names()[i]}")
    initialize_obs_utils = True
    for i in tasks:
        try:
            dataset_path = os.path.join(datasets_path, benchmark.get_task_demonstration(i))
            task_i_dataset, shape_meta = get_dataset(dataset_path, obs_modality=obs_modality,
                                                    initialize_obs_utils=initialize_obs_utils, seq_len=max_traj_len)
        except Exception as e:
            print(f"{e}:[error] failed to load task {i} name {benchmark.get_task_names()[i]}")
        initialize_obs_utils = False
        task_i_dataset = clean_dataset(task_i_dataset, traj_lengths[i], task_info[i], i, trajectories)
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(LANG_EMBEDDING_CFG, descriptions)
    benchmark.set_task_embs(task_embs)
    datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {len(tasks)}")
    for i in tasks:
        print(f"    - Task {i + 1}:")
        print(f"        {benchmark.get_task(i).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    concat_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            num_workers = 0,
            sampler = RandomSampler(concat_dataset),
            persistent_workers = False)
    return dataloader
