# Learning Parameterized Skills from Demonstrations (NeurIPS 2025)

This repository contains the official implementation of DEPS, an end-to-end algorithm for discovering parameterized skills from expert demonstrations. DEPS jointly learns discrete skills, their continuous parameters, and corresponding low-level action policies through a hierarchical architecture, combining the structure of discrete skills with the flexibility of continuous conditioning.


paper: link coming soon!

website: https://sites.google.com/view/parameterized-skills


## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.3+ (for GPU support)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/guptbot/DEPS.git
   cd DEPS
   ```

2. **Create and activate virtual environment**
   ```bash
   python3.9 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install PyTorch with CUDA 11.3**
   ```bash
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

4. **Install all dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will automatically install LIBERO from GitHub along with all other required packages.

5. **Configure LIBERO path**

   Update the `libero_root_path` in your command or in `config.py`:
   ```bash
   --libero_root_path /path/to/libero/directory/
   ```

6. **Download LIBERO datasets**

   Follow instructions at https://github.com/Lifelong-Robot-Learning/LIBERO to download the required datasets (e.g., LIBERO-100). Make sure the downloaded datasets are stored in `libero/libero/datasets`

## Usage

NOTE: the main branch of this repository contains the implementation of DEPS for use with LIBERO. To use DEPS with MetaWorld, checkout branch `metaworld`.

1. Pretraining:

    To pretrain DEPS:
    ```
    python main.py --train 1 --name run_name --batch_size 3 --seed 95 --markov_subpolicy 1 --train_dataset libero_90 --test_dataset libero_90  --algo deps --apply_aug 1 --number_policies 20 --hidden_size 1024 --hidden_size_primitive 1024 --hidden_size_compression 1024 --epochs 21 --z_dimensions 4 --kl_loss_weight 0.5
    ```

2. Finetuning:

    To finetune the checkpoint from step (1), run:
    ```
    python main.py --finetune 1 --name run_name --batch_size 2 --seed 95 --markov_subpolicy 1 --train_dataset libero_90 --test_dataset libero_90 --algo deps --apply_aug 0 --number_policies 20 --finetune_subpolicy 1 --finetune_resnet 1 --kl_loss_weight 0 --z_loss_weight 0 --hidden_size 1024 --hidden_size_primitive 1024 --hidden_size_compression 1024 --epochs 20 --z_dimensions 4 --model /path/to/saved/checkpoint --use_corrected_action 0 --downstream_rollout_frequ 2
    ```

More details on the specific commands to recreate the various presented runs in the paper can be found in  `run_commands.md`. 

In our experiments, we perform pretraining on 2 `NVIDIA GeForce RTX 3090` GPUs and finetuning on a single `NVIDIA GeForce RTX 3090` GPU. For MetaWorld, we use a single GPU for both pretraining and finetuning. 


## Provided LfD Algorithms

- **deps**: DEPS with state compression - learns 1D compressed robot state representation
- **deps_uncompressed**: DEPS without compression - uses full robot state
- **multitask**: Baseline that trains a single policy for all tasks

## Directory Structure

```
├── main.py                    # Entry point
├── config.py                  # Argument parser
├── train.py                   # Base training class (PolicyManagerBase)
├── deps.py                    # DEPS with compression
├── deps_uncompressed.py       # DEPS without compression
├── multitask.py               # Multitask baseline
├── networks/
│   ├── networks.py            # Core neural architectures
│   ├── gmm_head.py            # GMM action head
│   └── resnet.py              # ResNet encoder with FiLM
├── env/
│   ├── libero_dataloader.py  # Dataset loading
│   ├── libero_rollout.py     # Environment wrapper
│   └── libero_spatial_encode.py  # Observation encoding
└── utils/
    ├── utils.py               # Helper functions
    ├── logger.py              # Logging interface (WandB)
    └── visualization_tools.py # Visualization utilities
```

## Citation 

If you find this repository useful, please cite our work:
```
@inproceedings{vedant2025deps,
  title     = {Learning parameterized skills from demonstrations},
  author    = {Vedant Gupta and Haotian Fu and Calvin Luo and Yiding Jiang and George Konidaris},
  booktitle = {NeurIPS},
  year      = {2025},
}
```
