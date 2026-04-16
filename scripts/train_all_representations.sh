#!/usr/bin/env bash

set -euo pipefail

# Runs a training matrix over:
# - representations: pixel, latent_vae
# - backbones: dit, cnn, mlp
# - conditioning: cfg, none
# - simulators: ode, sde
#
# Defaults are tuned for a single GPU. Override any variable inline, e.g.
# GPU=2 LR=1e-3 VAE_CKPT=runs/vae/checkpoints/step_004999_model.pt bash scripts/train_all_representations.sh

GPU="${GPU:-2}"
DEVICE="${DEVICE:-cuda}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
LR="${LR:-1e-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VAE_CKPT="${VAE_CKPT:-runs/vae/checkpoints/step_004999_model.pt}"
BACKBONES=(dit cnn mlp)
CONDITIONINGS=(cfg none)
SIMULATORS=(ode sde)
REPRESENTATIONS=(pixel latent_vae)

for representation in "${REPRESENTATIONS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for conditioning in "${CONDITIONINGS[@]}"; do
      for simulator in "${SIMULATORS[@]}"; do
        run_name="${backbone}_${conditioning}_${simulator}_${representation}"

        cmd=(
          env
          "PYTHONPATH=${PYTHONPATH_VALUE}"
          "CUDA_VISIBLE_DEVICES=${GPU}"
          python
          scripts/train.py
          task=flow_matching
          training.device="${DEVICE}"
          representation="${representation}"
          backbone="${backbone}"
          conditioning="${conditioning}"
          simulator="${simulator}"
          training.run_name="${run_name}"
          training.lr="${LR}"
        )

        if [[ "${representation}" == "latent_vae" ]]; then
          cmd+=(
            representation.vae_checkpoint_path="${VAE_CKPT}"
          )
        fi

        if [[ -n "${EXTRA_ARGS}" ]]; then
          # Intentionally allow word splitting so Hydra overrides can be passed through.
          # shellcheck disable=SC2206
          extra_parts=(${EXTRA_ARGS})
          cmd+=("${extra_parts[@]}")
        fi

        echo "============================================================"
        echo "Starting run: ${run_name}"
        echo "GPU=${GPU} DEVICE=${DEVICE} LR=${LR}"
        if [[ "${representation}" == "latent_vae" ]]; then
          echo "VAE checkpoint: ${VAE_CKPT}"
        fi
        echo "Command: ${cmd[*]}"
        echo "============================================================"

        "${cmd[@]}"

        echo "Finished run: ${run_name}"
        echo
        sleep 2
      done
    done
  done
done

echo "All 24 runs completed."
