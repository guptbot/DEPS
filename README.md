# Learning parameterized skills from demonstrations

Installation instructions and detailed run commands for metaworld coming soon!

#### Example Usage ####
```
python main.py --train 1 --name test --batch_size 2 --algo deps --debug 1 --markov_subpolicy 1
```
Key args: 
 - set train/finetune to 1 to train/finetune the network
 - use algo to change which algorithm to use. [multitask: for a multitask baseline, deps_uncompressed: the standard parameterized skill algorithm we have been using, deps: a new alternative algorithm that learns parameterized **trajectories** by compressing the state to 1D]
 - see config.py for all other args

Key Files:
 - main.py: the entry point
 - train.py: implements the class for training and finetuning
 - [multitask.py, deps_uncompressed.py, deps.py]: inherits the base class to implement specific algorithms.
 - networks.py: contains most networks used (except resnet, gmm_head)


