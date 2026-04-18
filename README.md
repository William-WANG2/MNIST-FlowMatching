# MNIST Flow Matching

This is the course project for COMP5421 Computer Vision. The repo can also serve as Pedagogical material for anyone who wishes to learn the theory and implemention of flow matching and diffusion model.

Config-driven MNIST training and sampling for flow matching with interchangeable backbones, conditioning modes, and simulators.

The repo supports:

- backbones: `mlp`, `cnn`, `dit`
- conditioning modes: `none`, `cfg`
- simulators: `ode`, `sde`
- tasks: `flow_matching`, `vae`
- representations: `pixel`, `latent_vae`
- dataset: MNIST only

## What This Repo Does

This project trains a vector field model on MNIST using a Gaussian probability path. The same learned model can then be sampled with either:

- an ODE solver for deterministic generation
- an SDE solver for stochastic generation, where the score model is derived from the vector field model via the Gaussian SDE Extension Trick.

When `conditioning=cfg`, the model uses classifier-free guidance with 11 label slots:

- `0` to `9`: digit labels
- `10`: the unconditional/null label

When `conditioning=none`, the model uses time conditioning only and ignores class labels.

The repo also includes a VAE model path that shares the same Hydra-driven training entrypoint. After a VAE is trained, the same flow/diffusion pipeline can be run either:

- directly in pixel space with `representation=pixel`
- in VAE latent space with `representation=latent_vae`, where images are encoded before training and sampled latents are decoded back to image space

## Repository Layout

```text
configs/
  backbone/{mlp,cnn,dit}.yaml
  conditioning/{none,cfg}.yaml
  data/mnist.yaml
  path/gaussian.yaml
  representation/{pixel,latent_vae}.yaml
  simulator/{ode,sde}.yaml
  task/{flow_matching,vae}.yaml
  training/default.yaml
  sampling/default.yaml
  vae/default.yaml
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

VAE-specific code lives in:

- `src/models/vae.py`
- `src/objectives/vae.py`
- `src/engine/vae_trainer.py`
- `src/objectives/latent_flow_matching.py`
- `src/engine/latent_flow_trainer.py`
- `src/latent_vae.py`

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

The training loader resizes images to `32x32`, converts them to tensors, and normalizes them using the values from `configs/data/mnist.yaml`.

## How Training Is Wired

The training path is:

1. `scripts/train.py` loads the Hydra config.
2. `src/run_train.py` builds the dataloader and dispatches on both `cfg.task.name` and `cfg.representation.name`.
3. `src/factories.py` selects the concrete components from the chosen config groups.
4. `src/engine/trainer.py` runs optimization and saves checkpoints for pixel-space flow matching.
5. `src/engine/latent_flow_trainer.py` runs latent-space flow matching when `representation=latent_vae`.
6. `src/engine/vae_trainer.py` provides VAE-specific reconstruction logging when `task=vae`.

The sampling path is:

1. `scripts/sample.py` loads the Hydra config and a checkpoint.
2. `src/run_sample.py` rebuilds the same backbone and probability path in either pixel or latent space.
3. `src/simulation/sampler.py` starts from Gaussian noise and integrates with the selected ODE or SDE simulator.
4. `src/latent_vae.py` decodes sampled latents back to image space when `representation=latent_vae`.
5. `src/visualization.py` saves the output grid.

## Training

Run the default experiment:

```bash
python scripts/train.py
```

The default is defined in `configs/train.yaml`:

- backbone: `dit`
- conditioning: `cfg`
- simulator: `ode`
- task: `flow_matching`
- representation: `pixel`
- batch size: auto (`256` for pixel flow matching, `64` for latent-VAE flow matching and VAE training)
- steps: `20000`

Set `training.batch_size=<N>` to override the automatic batch-size selection explicitly.

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

### VAE Training

Train the VAE through the same entrypoint by switching the task:

```bash
python scripts/train.py task=vae training.run_name=vae
```

Example with common VAE overrides:

```bash
python scripts/train.py \
  task=vae \
  training.run_name=vae_beta2 \
  vae.hidden_channels='[16,32,64]' \
  vae.beta=2.0 \
  training.num_steps=5000 \
  training.batch_size=64 \
  training.lr=1e-3
```

This path uses:

- `src/models/vae.py` for the model definition
- `src/objectives/vae.py` for the objective wrapper
- `src/engine/vae_trainer.py` for optimization, checkpointing, and reconstruction visualizations

### Latent Flow/Diffusion Training With a VAE

After the VAE is trained, switch the representation from pixel space to latent space:

```bash
python scripts/train.py \
  task=flow_matching \
  representation=latent_vae \
  representation.vae_checkpoint_path=runs/vae/checkpoints/step_005000_model.pt \
  representation.latent_shape='[64,8,8]' \
  backbone=dit \
  conditioning=cfg \
  simulator=ode \
  training.run_name=dit_cfg_ode_latent
```

For representation-aware backbone defaults:

- latent `dit` uses `latent_patch_size=1` by default
- latent `cnn` disables encoder downsampling by default (`latent_downsample=false`) so 4x4 latent maps do not get spatially compressed further
- latent `cnn` defaults decoder upsampling ops to deconvolution when upsampling is enabled (`latent_upsample_mode=deconv`)
- latent `mlp` uses `latent_hidden_dims=[4096,4096,4096,4096]`, while pixel-space MLP keeps the original `pixel_hidden_dims=[1024,1024,1024]`
- set `backbone.patch_size`, `backbone.channels`, `backbone.num_residual_layers`, `backbone.mid_num_residual_layers`, `backbone.downsample`, or `backbone.upsample_mode` explicitly if you want to override those defaults

The latent training workflow is:

1. load the frozen VAE checkpoint
2. encode each MNIST image into a latent tensor
3. train the vector field model on the latent tensor instead of the pixel tensor
4. keep ODE/SDE simulation and CFG logic unchanged at the latent level
5. decode latent-space visualizations back to image space for logging

`representation=pixel` preserves the original behavior and does not involve the VAE at all.

### Useful Training Overrides

Change the number of steps:

```bash
python scripts/train.py training.num_steps=5000
```

Override learning rate and batch size explicitly:

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

For flow matching, each comparison image contains three rows:

- `target`: original training images
- `input (x_t)`: noisy path input at sampled time `t`
- `predicted target`: model-based reconstruction from the vector field prediction

For VAE training, each comparison image contains two rows:

- `target`: original training images
- `reconstruction`: the decoder output from the sampled latent

For latent-space flow matching, the logged comparison image still contains three rows, but the noisy input and prediction are first decoded from latent space back into image space.

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

This writes one image grid per guidance scale from `configs/sampling/default.yaml`.

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

### Using a Trained VAE With ODE and SDE Pipelines

A trained VAE checkpoint can now be used directly with the repo sampling pipeline:

```bash
python scripts/sample.py \
  representation=latent_vae \
  representation.vae_checkpoint_path=runs/vae/checkpoints/step_005000_model.pt \
  representation.latent_shape='[128,4,4]' \
  backbone=dit \
  conditioning=cfg \
  simulator=ode \
  sampling.checkpoint_path=runs/dit_cfg_ode_latent/checkpoints/step_010000_model.pt \
  sampling.output_dir=artifacts/dit_cfg_ode_latent
```

The same latent model can also be sampled with the SDE simulator:

```bash
python scripts/sample.py \
  representation=latent_vae \
  representation.vae_checkpoint_path=runs/vae/checkpoints/step_005000_model.pt \
  representation.latent_shape='[128,4,4]' \
  backbone=dit \
  conditioning=cfg \
  simulator=sde \
  sampling.checkpoint_path=runs/dit_cfg_sde_latent/checkpoints/step_010000_model.pt \
  sampling.output_dir=artifacts/dit_cfg_sde_latent
```

In latent sampling mode, `scripts/sample.py`:

1. samples Gaussian noise in the latent shape
2. simulates the latent trajectory with ODE or SDE dynamics
3. decodes the final latents with the trained VAE
4. writes image grids to the sampling output directory

## Full Workflow

If you want to use the VAE together with a diffusion or flow model, the intended workflow is:

1. Train the VAE with `task=vae`.
2. Save the VAE checkpoint from `runs/<vae_run_name>/checkpoints/`.
3. Train the flow/diffusion model with `task=flow_matching representation=latent_vae`.
4. Sample the latent model with `scripts/sample.py representation=latent_vae` and either `simulator=ode` or `simulator=sde`.
5. Decode the sampled latents automatically back to image space through the same VAE checkpoint.

There is still no single command that jointly trains the VAE and the latent flow model in one launch. The intended workflow is two-stage: VAE first, latent generative model second.

## Config Groups

The active experiment is composed from Hydra config groups.

Backbone configs in `configs/backbone`:

- `mlp.yaml`: hidden layer sizes and embedding sizes
- `cnn.yaml`: U-Net channel widths, residual depth, middle depth, optional downsampling, and upsampling mode (`bilinear` or `deconv`)
- `dit.yaml`: patch size, transformer depth, width, heads, and embedding sizes

Conditioning configs in `configs/conditioning`:

- `none.yaml`: unconditional generation
- `cfg.yaml`: classifier-free guidance with null label `10`

Simulator configs in `configs/simulator`:

- `ode.yaml`: Euler solver with deterministic trajectories
- `sde.yaml`: Euler-Maruyama solver with diffusion scale

Task configs in `configs/task`:

- `flow_matching.yaml`: image-space flow-matching training
- `vae.yaml`: VAE training

Representation configs in `configs/representation`:

- `pixel.yaml`: original image-space behavior
- `latent_vae.yaml`: latent-space flow/diffusion behavior backed by a trained VAE

Training config in `configs/training/default.yaml`:

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

Sampling config in `configs/sampling/default.yaml`:

- checkpoint path
- output directory
- samples per class
- guidance scales
- progress bar toggle

VAE config in `configs/vae/default.yaml`:

- hidden channel widths
- beta coefficient

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

For VAE runs, the comparison images in `runs/<run_name>/train_logs/comparisons/` are reconstruction grids.

## Notes

- The repo targets MNIST only.
- The probability path is Gaussian with linear `alpha` and `beta`.
- `conditioning=cfg` uses 11 label slots with the last slot reserved for the unconditional/null label.
- `conditioning=none` removes class conditioning from the backbone and sampling path.
- The simulator choice affects sampling only; training still learns the vector field defined by the flow-matching objective.
- `task=vae` reuses the same data loader and training configuration surface, but swaps in the VAE model, objective, and reconstruction logger.
- `representation=latent_vae` keeps the original flow/diffusion training logic but moves the model state space from pixels to VAE latents.
