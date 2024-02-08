#!/bin/bash
# if [ -z ${GPU} ]; then
#     GPU=0
# fi
# echo "Using GPU: ${GPU}"

GPU=4
DATA_DIR=summaries
export CUDA_VISIBLE_DEVICES=$GPU
# if [ -z ${DATA_DIR} ]; then
#     echo "DATA_DIR should be specified"
#     exit 1;
# fi
# CKPT_DIR=$(dirname $1)
# CKPT=$1
ESM_TYPE=esm2_t33_650M_UR50D
CKPT_DIR=logs/SKEMPI/${ESM_TYPE}/ita
BEST_ITER=17
CKPT=${CKPT_DIR}/mcatt_iter_${BEST_ITER}.ckpt
ESM_CKPT=${CKPT_DIR}/esm_iter_${BEST_ITER}.ckpt

echo "Using checkpoint: ${CKPT} and ${ESM_CKPT}"

python ita_generate.py \
    --ckpt ${CKPT} \
    --esm_ckpt ${ESM_CKPT} \
    --esm_type ${ESM_TYPE} \
    --test_set ${DATA_DIR}/skempi_all.json \
    --save_dir ${CKPT_DIR}/ita_results \
    --gpu 0 \
    --n_samples 100