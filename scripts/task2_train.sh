#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Paths to configuration files
cdr_config_path="${ROOT_DIR}/configs/datamodule/cdr.yaml"
cdr_design_config_path="${ROOT_DIR}/configs/experiment/fixedbb/cdr_design_esm2_650m.yaml"

# Check if the config files exist
if [[ ! -f $cdr_config_path ]] || [[ ! -f $cdr_design_config_path ]]; then
    echo "Config files not found. Exiting."
    exit 1
fi

# Backup original configuration files
cp ${cdr_config_path} ${cdr_config_path}.backup
cp ${cdr_design_config_path} ${cdr_design_config_path}.backup

# Array of alpha values
alphas=(0)

# pLM Model Name
# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D" "esm2_t33_650M_UR50D")
model_names=("esm2_t33_650M_UR50D")


for alpha in "${alphas[@]}"; do
    for model_name in "${model_names[@]}"; do
        exp=fixedbb/cdr_design_esm2_650m
        name=fixedbb/SKEMPI/${model_name}
        dataset=cdr

        # Run the experiment with the desired data_dir and cdr_type values directly in the command
        python ${ROOT_DIR}/train.py \
            experiment=${exp} \
            datamodule=${dataset} \
            name=${name} \
            logger=wandb \
            trainer=ddp_fp16 \
            datamodule.data_dir='${paths.data_dir}/' \
            task.learning.cdr_type=3 \
            logger.wandb.project=SKEMPI \
            task.learning.alpha=${alpha} \
            model.name=${model_name} \
            trainer.max_epochs=30 \
            
        # Restore original configuration files after each iteration
        [ -f ${cdr_config_path}.backup ] && cp ${cdr_config_path}.backup ${cdr_config_path}
        [ -f ${cdr_design_config_path}.backup ] && cp ${cdr_design_config_path}.backup ${cdr_design_config_path}
    done
done

# Cleanup backup files
[ -f ${cdr_config_path}.backup ] && rm ${cdr_config_path}.backup
[ -f ${cdr_design_config_path}.backup ] && rm ${cdr_design_config_path}.backup
