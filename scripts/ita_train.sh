#!/bin/bash

# Root directory
ROOT_DIR=$(pwd)
echo "Locate project at ${ROOT_DIR}"



if [ -z ${LR} ]; then
    LR=1e-3
fi

GPU=4
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: $GPU"

CKPT_DIR=summaries/ckpt/mean_CDR3_111/version_0_SKEMPI
DATA_DIR=${CKPT_DIR}/../../..
CKPT=${CKPT_DIR}/checkpoint/best.ckpt

ESM_TYPE=esm2_t33_650M_UR50D
ESM_CKPT=${ROOT_DIR}/logs/SKEMPI/${ESM_TYPE}/checkpoints/best.ckpt

echo "Using checkpoint: ${CKPT} and ESM checkpoint: ${ESM_CKPT}"

python ita_train.py \
    --pretrain_ckpt ${CKPT} \
    --esm_ckpt ${ESM_CKPT} \
    --esm_type ${ESM_TYPE} \
    --test_set ${DATA_DIR}/skempi_all.json \
    --save_dir ${ROOT_DIR}/logs/SKEMPI/${ESM_TYPE}/ita/ \
    --batch_size 4 \
    --update_freq 4 \
    --gpu 0 \
    --lr ${LR}
