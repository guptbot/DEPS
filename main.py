import torch
from config import *
from env.metaworld_dataloader import fetch_dataloader
from deps_uncompressed import DepsUncompressed
from multitask import Multitask
from deps import Deps
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALGO_TO_POLICY_MANAGER = {"deps_uncompressed": DepsUncompressed, "multitask": Multitask, "deps": Deps}
TEST_SETS = {
    "mw-vanilla": [45, 46, 47, 48, 49],
    "mw-prise": [10, 23, 24, 27, 37]
}

def main(args):
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()
    if args.algo not in ALGO_TO_POLICY_MANAGER:
        raise Exception("Unknown algorithm {}".format(args.algo))
    datasets_path = args.dataset_folder
    test_tasks = TEST_SETS[args.metaworld_test_set]
    train_tasks = [i for i in range(10)]
    args.test_tasks = test_tasks
    if args.finetune:

        num_tasks = 1 if args.debug else len(test_tasks)
        if args.few_shot_finetune: train_trajs = list(range(3))
        else: train_trajs = list(range(50))
        #valid_trajs = list(range(45, 50))
        train_dataloaders = []
        #valid_dataloaders = []
        for task_idx in test_tasks:
            train_dataloader = fetch_dataloader(datasets_path, args.batch_size, [task_idx], train_trajs, args.traj_length)
            #valid_dataloader = fetch_dataloader(datasets_path, args.test_dataset, 1, [task_idx], valid_trajs, args.traj_length)
            train_dataloaders.append(train_dataloader)
            #valid_dataloaders.append(valid_dataloader)
        torch.cuda.empty_cache()
        policy_manager = ALGO_TO_POLICY_MANAGER[args.algo](args=args)
        policy_manager.finetune(num_tasks, train_dataloaders)
        return
    train_trajs = list(range(50))
    #validation_trajs = list(range(45, 50))

    test_trajs = list(range(50))
    log_tasks = [0, 1, 2]
    if args.debug:
        train_tasks, test_tasks = [1], [1]
    print("training on tasks", train_tasks)
    train_dataloader = fetch_dataloader(datasets_path, args.batch_size, train_tasks, train_trajs, args.traj_length)
    #train_dataloader = fetch_dataloader(datasets_path, args.batch_size, [0], [0], args.traj_length)
    #valid_dataloader = fetch_dataloader(datasets_path, args.train_dataset, 1, train_tasks, validation_trajs, args.traj_length)
    test_dataloader = fetch_dataloader(datasets_path, 1, test_tasks, test_trajs, args.traj_length)
    task_dataloader = fetch_dataloader(datasets_path, args.batch_size, log_tasks, train_trajs, args.traj_length)
    torch.cuda.empty_cache()
    policy_manager = ALGO_TO_POLICY_MANAGER[args.algo](dataloader=train_dataloader, test_dataloader = test_dataloader, task_dataloader = task_dataloader, args=args)
    policy_manager.train()

if __name__ == '__main__':
    main(sys.argv)
