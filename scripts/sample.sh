#!/bin/bash

# Define the lists of options
backbones=("dit" "cnn" "mlp")
conditionings=("none" "cfg")
simulators=("ode" "sde")

# Loop over all combinations
for backbone in "${backbones[@]}"; do
  for conditioning in "${conditionings[@]}"; do
    for simulator in "${simulators[@]}"; do
      # Construct the checkpoint path and output dir
      run_dir="runs/${backbone}_${conditioning}_${simulator}"
      checkpoint_path="${run_dir}/checkpoints/step_020000_model.pt"
      output_dir="artifacts/${backbone}_${conditioning}_${simulator}"
      
      # Run the command
      CUDA_VISIBLE_DEVICES=7 python scripts/sample.py \
        backbone=${backbone} \
        conditioning=${conditioning} \
        simulator=${simulator} \
        sampling.checkpoint_path=${checkpoint_path} \
        sampling.output_dir=${output_dir}
      
      echo "Completed: backbone=${backbone}, conditioning=${conditioning}, simulator=${simulator}"
    done
  done
done
