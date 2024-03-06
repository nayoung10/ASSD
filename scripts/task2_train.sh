#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# pLM Model Name
model_name="esm2_t33_650M_UR50D"

exp=esm2
name=rabd/${model_name}
dataset=cdr

# Run the experiment with the desired data_dir and cdr_type values directly in the command
python ${ROOT_DIR}/train.py \
    experiment=${exp} \
    datamodule=${dataset} \
    name=${name} \
    trainer=ddp_fp16 \
    datamodule.data_dir='${paths.data_dir}/cdrh3' \
    task.learning.cdr_type=3 \
    model.name=${model_name} \
    trainer.max_epochs=30 \
