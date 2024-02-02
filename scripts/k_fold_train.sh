#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Paths to configuration files
cdr_config_path="${ROOT_DIR}/configs/datamodule/cdr.yaml"
cdr_design_config_path="${ROOT_DIR}/configs/experiment/fixedbb/cdr_design_esm1b_650m.yaml"

# Check if the config files exist
if [[ ! -f $cdr_config_path ]] || [[ ! -f $cdr_design_config_path ]]; then
    echo "Config files not found. Exiting."
    exit 1
fi

# Array of CDR types and folds
alphas=(0.1 0.125 0.15 0.175 0.2 0.8 1 2 3 4 5)
cdr_types=(3)
# folds=(0 1 2 3 4 5 6 7 8 9)
folds=(0)

# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
model_names=("esm2_t6_8M_UR50D")

# Main loop
for alpha in "${alphas[@]}"; do
    for model_name in "${model_names[@]}"; do
        for cdr in "${cdr_types[@]}"; do
            for fold in "${folds[@]}"; do

                exp=fixedbb/cdr_design_esm1b_650m
                name="fixedbb/cdrh${cdr}/fold_${fold}/${model_name}_RL_alpha${alpha}_LoRA"
                dataset=cdr

                # Run the experiment, passing the data_dir and cdr_type values directly
                python ${ROOT_DIR}/train.py \
                    experiment=${exp} \
                    datamodule=${dataset} \
                    name=${name} \
                    logger=wandb \
                    trainer=ddp_fp16 \
                    datamodule.data_dir='${paths.data_dir}/cdrh'"${cdr}"'/fold_'"${fold}" \
                    task.learning.cdr_type=${cdr} \
                    logger.wandb.project=task1_${model_name}_RL \
                    task.learning.alpha=${alpha} \
                    model.name=${model_name} \
                    trainer.max_epochs=30 \
                    task.learning.RL=true \

            done
        done
    done
done