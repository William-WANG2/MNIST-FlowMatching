#!/usr/bin/env bash

# run_combinations_gpu7.sh

GPU=7
SCRIPT="python scripts/train.py"

for backbone in cnn dit mlp; do
    for conditioning in cfg none; do
        for simulator in ode sde; do
            run_name="${backbone}_${conditioning}_${simulator}"
            
            echo "──────────────────────────────────────────────"
            echo "Starting run: ${run_name}"
            echo "GPU: ${GPU}"
            echo "──────────────────────────────────────────────"
            
            CUDA_VISIBLE_DEVICES=${GPU} ${SCRIPT} \
                backbone=${backbone} \
                conditioning=${conditioning} \
                simulator=${simulator} \
                training.run_name=${run_name}
            
            echo ""
            echo "Finished: ${run_name}"
            echo ""
            
            # Optional: small pause to let GPU cool / logs flush
            sleep 3
        done
    done
done

echo "All 12 combinations completed."