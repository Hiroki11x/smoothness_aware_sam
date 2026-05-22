# SAM Calibration

Research code for calibration and out-of-distribution experiments with:

- `SAM`
- `SA-SGD`
- `SA-SAM`
- standard baselines such as `SGD`, `Momentum SGD`, `Adam`, and `AdamW`

The main `src/` implementation has been cleaned up to match the paper version of SA-SAM more closely:

- `sasgd` is the canonical smoothness-adaptive SGD name
- `sasam` is the canonical smoothness-adaptive SAM name
- legacy aliases `adsgd` and `adsgd_sam` are still accepted for backward compatibility
- SA-SAM uses the paper-style adaptive perturbation radius `rho_t = sqrt(eta_t)`

## Repository Layout

- `src/`: main training pipeline for calibration and OOD experiments
- `tests/`: lightweight regression tests for optimizer behavior and runtime configuration
- `exp/`: small generic utilities and notes; cluster-specific launchers were removed

## Installation

Use a Python environment with PyTorch installed, then install the project dependencies:

```sh
pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes the `ImageNetV2_pytorch` dependency used for ImageNet-V2 evaluation.
- `numpy<2` is pinned in `requirements.txt` because the current PyTorch environment in this project is not yet compatible with NumPy 2 in all setups.
- `backpack-for-pytorch` is optional. Some experimental code checks for it, but the main training path does not require it.

## Main Training Entry Point

The primary entry point is:

```sh
python src/main.py
```

Example: CIFAR-10 with paper-style SA-SAM

```sh
python src/main.py \
  --dataset cifar10 \
  --ood_dataset cifar10_1 \
  --model resnet18_1_cifar \
  --opt sasam \
  --lr 0.05 \
  --batch_size 256 \
  --epochs_budget 200 \
  --wandb_project_name sam-calibration \
  --wandb_offline
```

Example: vanilla SAM baseline

```sh
python src/main.py \
  --dataset cifar10 \
  --ood_dataset cifar10_1 \
  --model resnet18_1_cifar \
  --opt sam \
  --rho 0.05 \
  --lr 0.05 \
  --batch_size 256 \
  --epochs_budget 200 \
  --wandb_offline
```

Supported optimizer names in `src/main.py` include:

- `vanilla_sgd`
- `momentum_sgd`
- `nesterov_momentum_sgd`
- `adam`
- `adamw`
- `sam`
- `sasgd`
- `sasam`

## Runtime Configuration

The code no longer depends on cluster-specific environment names such as Mila or Narval.

Optional environment variables:

- `SAM_CALIBRATION_CUDA_VISIBLE_DEVICES`: explicitly sets `CUDA_VISIBLE_DEVICES`
- `SAM_CALIBRATION_DATA_ROOT`: overrides the dataset root for all datasets
- `IMAGENET_DATA_ROOT`: fallback root for ImageNet when `--data_root` is left at its default
- `LOCAL_SCRATCH_DIR`, `SLURM_TMPDIR`, `TMPDIR`, `DATA_DIR_PATH`: optional scratch-directory hints for helper utilities

W&B configuration is now generic:

- `--wandb_entity` is optional
- if omitted, runs are created without a hard-coded team or organization

## Tests

Run the lightweight regression tests with:

```sh
python -m unittest tests.test_sasam_paper_behavior tests.test_misc_runtime_config
```

These tests currently cover:

- canonical optimizer aliases
- SA-SAM adaptive `rho` behavior
- generic runtime environment and dataset-root resolution

## Removed Cluster-Specific Assets

Historical cluster launchers and environment setup files were removed from the working tree during cleanup because they contained:

- hard-coded usernames
- hard-coded absolute paths
- site-specific scheduler assumptions
- site-specific W&B defaults

If you need batch-job launchers again, create new generic scripts on top of the documented CLI examples above instead of reintroducing cluster-specific paths into the repository.
