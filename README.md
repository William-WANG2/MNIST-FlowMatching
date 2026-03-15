# MNIST Flow Matching

Config-driven MNIST training and sampling for flow matching with interchangeable backbones, conditioning modes, and simulators.

The repo supports:

- backbones: `mlp`, `cnn`, `dit`
- conditioning modes: `none`, `cfg`
- simulators: `ode`, `sde`
- dataset: MNIST only

## What This Repo Does

This project trains a vector field model on MNIST using a Gaussian probability path. The same learned model can then be sampled with either:

- an ODE solver for deterministic generation
- an SDE solver for stochastic generation

When `conditioning=cfg`, the model uses classifier-free guidance with 11 label slots:

- `0` to `9`: digit labels
- `10`: the unconditional/null label

When `conditioning=none`, the model uses time conditioning only and ignores class labels.

## Repository Layout

```text
configs/
  backbone/{mlp,cnn,dit}.yaml
  conditioning/{none,cfg}.yaml
  data/mnist.yaml
  path/gaussian.yaml
  simulator/{ode,sde}.yaml
  training/default.yaml
  sampling/default.yaml
  train.yaml
  sample.yaml
scripts/
  train.py
  sample.py
src/
  data/
  engine/
  models/
  objectives/
  probability_paths/
  simulation/
  utils/
  factories.py
  run_train.py
  run_sample.py
  visualization.py
```

## Conda Setup

Create the environment from the repo root:

```bash
conda env create -f environment.yml
conda activate mnist-flow-matching
```

The environment file installs the package in editable mode, so local code changes are picked up immediately.

If you already created the environment and want to refresh it:

```bash
conda env update -f environment.yml --prune
conda activate mnist-flow-matching
```

If you need a custom PyTorch build for a specific CUDA setup, install the appropriate `pytorch` and `torchvision` packages first, then run:

```bash
pip install -e .
```

## Data

MNIST is downloaded automatically on first training run into `data/`:

```text
data/MNIST/
```

The training loader resizes images to `32x32`, converts them to tensors, and normalizes them using the values from [configs/data/mnist.yaml](/Users/wangxuezhen/Downloads/MNIST/configs/data/mnist.yaml).

## How Training Is Wired

The training path is:

1. [scripts/train.py](/Users/wangxuezhen/Downloads/MNIST/scripts/train.py) loads the Hydra config.
2. [src/run_train.py](/Users/wangxuezhen/Downloads/MNIST/src/run_train.py) builds the dataloader, probability path, backbone, and objective.
3. [src/factories.py](/Users/wangxuezhen/Downloads/MNIST/src/factories.py) selects the concrete components from the chosen config groups.
4. [src/engine/trainer.py](/Users/wangxuezhen/Downloads/MNIST/src/engine/trainer.py) runs optimization and saves checkpoints.

The sampling path is:

1. [scripts/sample.py](/Users/wangxuezhen/Downloads/MNIST/scripts/sample.py) loads the Hydra config and a checkpoint.
2. [src/run_sample.py](/Users/wangxuezhen/Downloads/MNIST/src/run_sample.py) rebuilds the same backbone and path.
3. [src/simulation/sampler.py](/Users/wangxuezhen/Downloads/MNIST/src/simulation/sampler.py) starts from Gaussian noise and integrates with the selected simulator.
4. [src/visualization.py](/Users/wangxuezhen/Downloads/MNIST/src/visualization.py) saves the output grid.

## Training

Run the default experiment:

```bash
python scripts/train.py
```

The default is defined in [configs/train.yaml](/Users/wangxuezhen/Downloads/MNIST/configs/train.yaml):

- backbone: `dit`
- conditioning: `cfg`
- simulator: `ode`
- batch size: `256`
- steps: `20000`

### Common Training Commands

Default DiT + CFG + ODE:

```bash
python scripts/train.py training.run_name=dit_cfg_ode
```

MLP + unconditional + ODE:

```bash
python scripts/train.py backbone=mlp conditioning=none simulator=ode training.run_name=mlp_none_ode
```

MLP + CFG + SDE:

```bash
python scripts/train.py backbone=mlp conditioning=cfg simulator=sde training.run_name=mlp_cfg_sde
```

CNN + unconditional + ODE:

```bash
python scripts/train.py backbone=cnn conditioning=none simulator=ode training.run_name=cnn_none_ode
```

CNN + CFG + SDE:

```bash
python scripts/train.py backbone=cnn conditioning=cfg simulator=sde training.run_name=cnn_cfg_sde
```

DiT + unconditional + ODE:

```bash
python scripts/train.py backbone=dit conditioning=none simulator=ode training.run_name=dit_none_ode
```

DiT + CFG + SDE:

```bash
python scripts/train.py backbone=dit conditioning=cfg simulator=sde training.run_name=dit_cfg_sde
```

### Useful Training Overrides

Change the number of steps:

```bash
python scripts/train.py training.num_steps=5000
```

Change learning rate and batch size:

```bash
python scripts/train.py training.lr=1e-4 training.batch_size=128
```

Force CPU:

```bash
python scripts/train.py training.device=cpu
```

Save to a custom run directory name:

```bash
python scripts/train.py training.run_name=my_experiment
```

Control periodic training visual logging (local files):

```bash
python scripts/train.py \
  training.image_log_every=250 \
  training.loss_curve_every=50 \
  training.compare_batch_size=16
```

### Training Logging

Training logs are file-based and include only training signals (no validation loop).

- `training.log_every`: tqdm text updates for `step/lr/loss`
- `training.image_log_every`: save periodic image comparisons
- `training.loss_curve_every`: refresh `loss_curve.png`
- `training.compare_batch_size`: number of images used in each comparison grid

Each comparison image contains three rows:

- `target`: original training images
- `input (x_t)`: noisy path input at sampled time `t`
- `predicted target`: model-based reconstruction from the vector field prediction

Note: `training.wandb.*` keys are not part of this config yet. Use the local file logs above.

## Sampling

Sampling requires a trained checkpoint. Checkpoints are written to:

```text
runs/<run_name>/checkpoints/
```

Model checkpoints are named like:

```text
step_001000_model.pt
```

### Sample From a CFG Model

```bash
python scripts/sample.py \
  backbone=dit \
  conditioning=cfg \
  simulator=ode \
  sampling.checkpoint_path=runs/dit_cfg_ode/checkpoints/step_010000_model.pt \
  sampling.output_dir=artifacts/dit_cfg_ode
```

This writes one image grid per guidance scale from [configs/sampling/default.yaml](/Users/wangxuezhen/Downloads/MNIST/configs/sampling/default.yaml).

Override the guidance scales:

```bash
python scripts/sample.py \
  backbone=dit \
  conditioning=cfg \
  simulator=ode \
  sampling.checkpoint_path=runs/dit_cfg_ode/checkpoints/step_010000_model.pt \
  sampling.guidance_scales='[1.0,2.0,4.0,6.0]'
```

### Sample From an Unconditional Model

```bash
python scripts/sample.py \
  backbone=cnn \
  conditioning=none \
  simulator=ode \
  sampling.checkpoint_path=runs/cnn_none_ode/checkpoints/step_010000_model.pt \
  sampling.output_dir=artifacts/cnn_none_ode
```

### Sample With the SDE Simulator

```bash
python scripts/sample.py \
  backbone=mlp \
  conditioning=cfg \
  simulator=sde \
  sampling.checkpoint_path=runs/mlp_cfg_sde/checkpoints/step_010000_model.pt \
  sampling.output_dir=artifacts/mlp_cfg_sde
```

## Config Groups

The active experiment is composed from Hydra config groups.

Backbone configs in [configs/backbone](/Users/wangxuezhen/Downloads/MNIST/configs/backbone):

- `mlp.yaml`: hidden layer sizes and embedding sizes
- `cnn.yaml`: U-Net channel widths, residual depth, and embedding sizes
- `dit.yaml`: patch size, transformer depth, width, heads, and embedding sizes

Conditioning configs in [configs/conditioning](/Users/wangxuezhen/Downloads/MNIST/configs/conditioning):

- `none.yaml`: unconditional generation
- `cfg.yaml`: classifier-free guidance with null label `10`

Simulator configs in [configs/simulator](/Users/wangxuezhen/Downloads/MNIST/configs/simulator):

- `ode.yaml`: Euler solver with deterministic trajectories
- `sde.yaml`: Euler-Maruyama solver with diffusion scale

Training config in [configs/training/default.yaml](/Users/wangxuezhen/Downloads/MNIST/configs/training/default.yaml):

- seed
- device
- batch size
- number of steps
- learning rate
- weight decay
- warmup steps
- logging frequency
- checkpoint frequency
- image log frequency
- loss curve frequency
- comparison batch size

Sampling config in [configs/sampling/default.yaml](/Users/wangxuezhen/Downloads/MNIST/configs/sampling/default.yaml):

- checkpoint path
- output directory
- samples per class
- guidance scales
- progress bar toggle

## Outputs

Training writes:

- `runs/<run_name>/config.yaml`
- `runs/<run_name>/checkpoints/step_XXXXXX_model.pt`
- `runs/<run_name>/checkpoints/step_XXXXXX_optimizer.pt`
- `runs/<run_name>/train_logs/loss_curve.png`
- `runs/<run_name>/train_logs/comparisons/step_XXXXXX.png`

Sampling writes:

- `artifacts/<experiment_name>/sample_config.yaml`
- `artifacts/<experiment_name>/*.png`

## Notes

- The repo targets MNIST only.
- The probability path is Gaussian with linear `alpha` and `beta`.
- `conditioning=cfg` uses 11 label slots with the last slot reserved for the unconditional/null label.
- `conditioning=none` removes class conditioning from the backbone and sampling path.
- The simulator choice affects sampling only; training still learns the vector field defined by the flow-matching objective.
