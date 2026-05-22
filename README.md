# Smoothness-Adaptive Sharpness-Aware Minimization

Reproduction-focused code for the ICLR 2024 paper:

- `Smoothness-Adaptive Sharpness-Aware Minimization for Finding Flatter Minima`

The repository is intentionally narrowed to the paper's core image-classification setup:

- training dataset: `CIFAR-10`
- OOD evaluation dataset: `CIFAR10-C`
- models: `vgg19`, `vit_small`
- optimizers: `momentum_sgd`, `adam`, `sam`, `sasgd`, `sasam`

Legacy aliases are still accepted:

- `adsgd` -> `sasgd`
- `adsgd_sam` -> `sasam`

`sasam` follows the paper-style adaptive perturbation radius:

- `rho_t = sqrt(eta_t)`

## Repository Layout

- `src/`: training, evaluation, optimizers, Hessian utilities, and models used in the paper
- `tests/`: regression tests for SA-SAM behavior and runtime configuration
- `exp/`: a minimal helper directory

## Installation

Use a Python environment with PyTorch installed, then install:

```sh
pip install -r requirements.txt
```

Notes:

- `numpy<2` is pinned because the current local PyTorch environment may be built against NumPy 1.x.
- `pyhessian` is only needed when `--calc_hessian` is enabled.

## Data Layout

Default training data root is `../data`.

Expected paths:

- `../data/cifar-10-batches-py` or the directory structure created by `torchvision.datasets.CIFAR10`
- `../data/CIFAR10_C/npy_files/*.npy` for CIFAR10-C

You can also pass a separate OOD root:

```sh
--ood_data_root /path/to/CIFAR10_C/npy_files
```

or its parent:

```sh
--ood_data_root /path/to/data
```

## Main Entry Point

The main entry point is:

```sh
python src/main.py
```

Example: paper-style SA-SAM on VGG-19

```sh
python src/main.py \
  --model vgg19 \
  --dataset cifar10 \
  --ood_dataset cifar10_c \
  --opt sasam \
  --lr 0.05 \
  --batch_size 256 \
  --epochs_budget 200 \
  --wandb_project_name smoothness-aware-sam \
  --wandb_offline
```

Example: vanilla SAM baseline on ViT-Small

```sh
python src/main.py \
  --model vit_small \
  --dataset cifar10 \
  --ood_dataset cifar10_c \
  --opt sam \
  --rho 0.05 \
  --lr 0.05 \
  --batch_size 256 \
  --epochs_budget 200 \
  --wandb_offline
```

## Hessian Metrics

To reproduce the flatness measurements from the paper, enable Hessian computation:

```sh
python src/main.py \
  --model vgg19 \
  --opt sasam \
  --calc_hessian \
  --calc_hessian_interval 25
```

This uses `pyhessian` to log:

- trace of the Hessian
- top eigenvalue of the Hessian
- optional gradient norm if `--calc_gradnorm`

## Tests

Run:

```sh
python -m unittest tests.test_sasam_paper_behavior tests.test_misc_runtime_config
```
