#!/usr/bin/env bash

set -euo pipefail

# Runs a training matrix over:
# - representations: pixel, latent_vae
# - backbones: dit, cnn, mlp
# - conditioning: cfg, none
# - simulators: ode, sde
#
# Defaults are tuned for a single GPU. Override any variable inline, e.g.
# GPU=2 LR=1e-3 VAE_NUM_STEPS=10000 bash scripts/train_all_representations.sh

GPU="${GPU:-2}"
DEVICE="${DEVICE:-cuda}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
LR="${LR:-1e-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VAE_RUN_NAME="${VAE_RUN_NAME:-vae}"
VAE_NUM_STEPS="${VAE_NUM_STEPS:-10000}"
VAE_EXTRA_ARGS="${VAE_EXTRA_ARGS:-}"
SAMPLE_EXTRA_ARGS="${SAMPLE_EXTRA_ARGS:-}"
BACKBONES=(dit cnn mlp)
CONDITIONINGS=(cfg none)
SIMULATORS=(ode sde)
REPRESENTATIONS=(pixel latent_vae)


latest_model_checkpoint() {
  local run_name="$1"
  local checkpoint_dir="runs/${run_name}/checkpoints"
  local checkpoint_path

  checkpoint_path="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'step_*_model.pt' | sort | tail -n 1)"
  if [[ -z "${checkpoint_path}" ]]; then
    echo "No model checkpoint found under ${checkpoint_dir}" >&2
    return 1
  fi

  printf '%s\n' "${checkpoint_path}"
}


append_override_args() {
  local extra_args="$1"
  if [[ -n "${extra_args}" ]]; then
    # Intentionally allow word splitting so Hydra overrides can be passed through.
    # shellcheck disable=SC2206
    local extra_parts=(${extra_args})
    printf '%s\0' "${extra_parts[@]}"
  fi
}


run_and_log() {
  local description="$1"
  shift

  echo "============================================================"
  echo "${description}"
  echo "Command: $*"
  echo "============================================================"
  "$@"
  echo
}


echo "Preparing VAE checkpoint for latent-space runs."

vae_train_cmd=(
  env
  "PYTHONPATH=${PYTHONPATH_VALUE}"
  "CUDA_VISIBLE_DEVICES=${GPU}"
  python
  scripts/train.py
  task=vae
  training.device="${DEVICE}"
  training.run_name="${VAE_RUN_NAME}"
  training.num_steps="${VAE_NUM_STEPS}"
)

if [[ -n "${VAE_EXTRA_ARGS}" ]]; then
  while IFS= read -r -d '' arg; do
    vae_train_cmd+=("${arg}")
  done < <(append_override_args "${VAE_EXTRA_ARGS}")
fi

run_and_log "Training VAE run ${VAE_RUN_NAME} on ${VAE_NUM_STEPS} steps" "${vae_train_cmd[@]}"

VAE_CKPT="$(latest_model_checkpoint "${VAE_RUN_NAME}")"

vae_sample_cmd=(
  env
  "PYTHONPATH=${PYTHONPATH_VALUE}"
  "CUDA_VISIBLE_DEVICES=${GPU}"
  python
  scripts/sample.py
  task=vae
  sampling.checkpoint_path="${VAE_CKPT}"
  sampling.output_dir="artifacts/${VAE_RUN_NAME}"
)

if [[ -n "${SAMPLE_EXTRA_ARGS}" ]]; then
  while IFS= read -r -d '' arg; do
    vae_sample_cmd+=("${arg}")
  done < <(append_override_args "${SAMPLE_EXTRA_ARGS}")
fi

run_and_log "Sampling VAE run ${VAE_RUN_NAME}" "${vae_sample_cmd[@]}"

echo "Using VAE checkpoint ${VAE_CKPT} for latent-space flow runs."
echo

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
          while IFS= read -r -d '' arg; do
            cmd+=("${arg}")
          done < <(append_override_args "${EXTRA_ARGS}")
        fi

        run_and_log "Training run ${run_name} (GPU=${GPU} DEVICE=${DEVICE} LR=${LR})" "${cmd[@]}"

        checkpoint_path="$(latest_model_checkpoint "${run_name}")"
        sample_cmd=(
          env
          "PYTHONPATH=${PYTHONPATH_VALUE}"
          "CUDA_VISIBLE_DEVICES=${GPU}"
          python
          scripts/sample.py
          task=flow_matching
          representation="${representation}"
          backbone="${backbone}"
          conditioning="${conditioning}"
          simulator="${simulator}"
          sampling.checkpoint_path="${checkpoint_path}"
          sampling.output_dir="artifacts/${run_name}"
        )

        if [[ "${representation}" == "latent_vae" ]]; then
          sample_cmd+=(
            representation.vae_checkpoint_path="${VAE_CKPT}"
          )
        fi

        if [[ -n "${SAMPLE_EXTRA_ARGS}" ]]; then
          while IFS= read -r -d '' arg; do
            sample_cmd+=("${arg}")
          done < <(append_override_args "${SAMPLE_EXTRA_ARGS}")
        fi

        run_and_log "Sampling run ${run_name}" "${sample_cmd[@]}"
        sleep 2
      done
    done
  done
done

echo "All 24 flow runs completed after VAE training and per-run sampling."
