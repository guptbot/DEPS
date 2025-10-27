import torch
from config import *
from env.libero_dataloader import fetch_dataloader
from deps_uncompressed import DepsUncompressed
from multitask import Multitask
from deps import Deps
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Map algorithm names to their corresponding policy manager classes
ALGO_TO_POLICY_MANAGER = {"deps_uncompressed": DepsUncompressed, "multitask": Multitask, "deps": Deps}

def main(args):
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()
    if args.algo not in ALGO_TO_POLICY_MANAGER:
        raise Exception("Unknown algorithm {}".format(args.algo))
    
    # Get the path to the LIBERO dataset
    datasets_path = args.libero_root_path + 'libero/libero/../datasets'

    # Finetune mode: fine-tune pretrained model on specific tasks
    if args.finetune:
        if args.test_dataset == 'libero_90':
            offset = args.lib90_split_idx # When lib90_split_idx = 80, this corresponds to the LIBERO-OOD test dataset.
        else:
            assert args.test_dataset == 'libero_10' # Use the LIBERO-10 dataset
            offset = 0

        num_tasks = 1 if args.debug else 10

        # LIBERO has 50 trajectories. Select which of these to finetunne on depending on whether or not we are doing 3-shot finetuning
        if args.three_shot_finetune: 
            train_trajs = list(range(3))
        else: 
            train_trajs = list(range(45))
        # Validation trajectories used to caluclate validation BC loss. 
        valid_trajs = list(range(45, 50))
        train_dataloaders = []
        valid_dataloaders = []
        for task_idx in range(offset, offset + num_tasks):
            train_dataloader = fetch_dataloader(datasets_path, args.test_dataset, args.batch_size, [task_idx], train_trajs, args.traj_length)
            valid_dataloader = fetch_dataloader(datasets_path, args.test_dataset, 1, [task_idx], valid_trajs, args.traj_length)
            train_dataloaders.append(train_dataloader)
            valid_dataloaders.append(valid_dataloader)
        torch.cuda.empty_cache()
        policy_manager = ALGO_TO_POLICY_MANAGER[args.algo](args=args)
        policy_manager.finetune(num_tasks, train_dataloaders, valid_dataloaders)
        return
    
    # Training mode: train model from scratch
    train_trajs = list(range(45))  # First 45 trajectories for training
    validation_trajs = list(range(45, 50))  # Trajectories 45-50 for logging validation BC loss

    test_trajs = list(range(50))  # All 50 trajectories for logging BC loss on the test dataset
    log_tasks = [46, 47, 66]  # Specific tasks to log for visualizations on skill decompositions being learned
    if args.debug:
        train_tasks, test_tasks = [1], [1]
    else:
        train_tasks = list(range(0, args.lib90_split_idx))
        if args.test_dataset == 'libero_90':
            test_tasks = list(range(args.lib90_split_idx, 90))
        else:
            assert args.test_dataset == "libero_10"
            test_tasks = list(range(10))

    train_dataloader = fetch_dataloader(datasets_path, args.train_dataset, args.batch_size, train_tasks, train_trajs, args.traj_length)
    valid_dataloader = fetch_dataloader(datasets_path, args.train_dataset, 1, train_tasks, validation_trajs, args.traj_length)
    test_dataloader = fetch_dataloader(datasets_path, args.test_dataset, 1, test_tasks, test_trajs, args.traj_length)
    task_dataloader = fetch_dataloader(datasets_path, args.train_dataset, args.batch_size, log_tasks, train_trajs, args.traj_length)
    torch.cuda.empty_cache()
    policy_manager = ALGO_TO_POLICY_MANAGER[args.algo](dataloader=train_dataloader, valid_dataloader = valid_dataloader, test_dataloader = test_dataloader, task_dataloader = task_dataloader, args=args)
    policy_manager.train()

if __name__ == '__main__':
    main(sys.argv)
