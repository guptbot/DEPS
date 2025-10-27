
### Pretraining 


To pretrain DEPS:

```
python main.py --train 1 --name run_name --batch_size 3 --seed 95 --markov_subpolicy 1 --train_dataset libero_90 --test_dataset libero_90  --algo deps --apply_aug 1 --number_policies 20 --hidden_size 1024 --hidden_size_primitive 1024 --hidden_size_compression 1024 --epochs 21 --z_dimensions 4 --kl_loss_weight 0.5
```

NOTE: the original experiments in Section 5 uses `number_policies = 10`. Appendix G suggests using `number_policies = 20` might lead to superior results. 

To pretrain DEPS without state compression:

```
python main.py --train 1 --name run_name --batch_size 3 --seed 95 --use_language 0 --markov_subpolicy 0 --train_dataset libero_90 --test_dataset libero_90 --image_to_subpolicy 0 --algo deps_uncompressed --apply_aug 1 --number_policies 10 --hidden_size 1024 --hidden_size_primitive 1024 --epochs 21
```

To pretrain the multitask baseline:

```
python main.py --train 1 --name run_name --batch_size 3 --seed 95 --use_language 1 --markov_subpolicy 0 --train_dataset libero_90 --test_dataset libero_90 --image_to_subpolicy 1 --algo multitask --apply_aug 1 --number_policies 2 --hidden_size 1024 --hidden_size_primitive 1024 --epochs 21 --gmm_hidden_size 512
```

### Finetuning

The commands below run finetuning on the LIBERO-OOD dataset. 

To instead evalute on LIBERO-10, set `test_dataset = libero_10`. 

To evaluate on LIBERO-3-shot, add `--tree_shot_finetune 1` to the command and also set `epochs = 250` and `downstream_rollout_freq = 25` (each `epoch` is now smaller as there are only three trajectories in the dataset.)

To finetune DEPS:

```
python main.py --finetune 1 --name run_name --batch_size 2 --seed 95 --use_language 0 --markov_subpolicy 1 --train_dataset libero_90 --test_dataset libero_90 --image_to_subpolicy 1 --algo deps --apply_aug 0 --number_policies 10 --finetune_subpolicy 1 --finetune_resnet 1 --kl_loss_weight 0 --z_loss_weight 0 --hidden_size 1024 --hidden_size_primitive 1024 --hidden_size_compression 1024 --epochs 20 --z_dimensions 4 --model /path/to/saved/checkpoint --use_corrected_action 0 --downstream_rollout_frequ 2
```

To finetune DEPS without state compression

```
python main.py --finetune 1 --name run_name --batch_size 2 --seed 95 --use_language 0 --markov_subpolicy 0 --train_dataset libero_90 --test_dataset libero_90 --image_to_subpolicy 0 --algo deps_uncompressed --apply_aug 0 --number_policies 10 --finetune_subpolicy 1 --finetune_resnet 1 --kl_loss_weight 0 --z_loss_weight 0 --hidden_size 1024 --hidden_size_primitive 1024 --epochs 21 --model /path/to/saved/checkpoint
```

To finetune the multitask baseline

```
python main.py --finetune 1 --name run_name --batch_size 2 --seed 95 --use_language 1 --markov_subpolicy 0 --train_dataset libero_90 --test_dataset libero_90 --image_to_subpolicy 1 --algo multitask --apply_aug 0 --number_policies 2 --finetune_subpolicy 1 --finetune_resnet 1 --kl_loss_weight 0 --z_loss_weight 0 --hidden_size 1024 --hidden_size_primitive 1024 --epochs 20 --model /path/to/saved/checkpoint --downstream_rollout_frequ 2 
```