#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Array of CDR types and folds
alphas=(0)
cdr_types=(1 2 3)
folds=(0 1 2 3 4 5 6 7 8 9)
# folds=(0)

# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
model_names=("esm2_t6_8M_UR50D")

export CUDA_VISIBLE_DEVICES=4
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

T=(3 4 5 6 7 8 9 10)
# Main loop for testing
for iter in "${T[@]}"; do
    for alpha in "${alphas[@]}"; do
        for model_name in "${model_names[@]}"; do
            for cdr in "${cdr_types[@]}"; do
                for fold in "${folds[@]}"; do
                    # Define experiment name dynamically
                    exp_name="fixedbb/cdrh${cdr}/fold_${fold}/${model_name}_full_finetune"

                    # Test 
                    exp_path="${ROOT_DIR}/logs/${exp_name}"
                    python ${ROOT_DIR}/test.py experiment_path=${exp_path} data_split=test ckpt_path=best.ckpt mode=predict task.generator.max_iter=${iter} task.generator.strategy='denoise'
                done
            done
        done
    done
done