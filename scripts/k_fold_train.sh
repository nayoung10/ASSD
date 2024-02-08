#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Array of CDR types and folds
cdr_types=(1 2 3)
folds=(0 1 2 3 4 5 6 7 8 9)

# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
model_name=("esm2_t33_650M_UR50D")

# Main loop
for cdr in "${cdr_types[@]}"; do
    for fold in "${folds[@]}"; do

        exp=esm2
        name="sabdab/cdrh${cdr}/fold_${fold}/${model_name}"
        dataset=cdr

        # Run the experiment, passing the data_dir and cdr_type values directly
        python ${ROOT_DIR}/train.py \
            experiment=${exp} \
            datamodule=${dataset} \
            name=${name} \
            trainer=ddp_fp16 \
            datamodule.data_dir='${paths.data_dir}/cdrh'"${cdr}"'/fold_'"${fold}" \
            task.learning.cdr_type=${cdr} \
            model.name=${model_name} \
            trainer.max_epochs=30 \

    done
done
