#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

cdr_types=(1 2 3)
folds=(0 1 2 3 4 5 6 7 8 9)

# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
model_name=("esm2_t33_650M_UR50D")

export CUDA_VISIBLE_DEVICES=4
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Main loop for testing
for cdr in "${cdr_types[@]}"; do
    for fold in "${folds[@]}"; do
        # Define experiment name dynamically
        exp_name="sabdab/cdrh${cdr}/fold_${fold}/${model_name}"

        # Test 
        exp_path="${ROOT_DIR}/logs/${exp_name}"
        python ${ROOT_DIR}/test.py experiment_path=${exp_path} data_split=test ckpt_path=best.ckpt mode=predict task.generator.max_iter=1 task.generator.strategy='denoise'
    done
done
