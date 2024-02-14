#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
if [ -z ${MODE} ]; then
    MODE=111
fi
if [ -z ${MODEL} ]; then
    MODEL=mean
fi
if [ -z ${CDR} ]; then
    CDR=3
fi

echo "Using GPU: ${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU

VERSION=0
if [ $1 ]; then
    VERSION=$1
fi

CKPT_DIR=${DATA_DIR}/ckpt/${MODEL}_CDR${CDR}_${MODE}/version_${VERSION}
CKPT=${CKPT_DIR}/checkpoint/best.ckpt

echo "Using checkpoint: ${CKPT}"

# model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
# train_types=("train_from_scratch" "full_finetune")

model_name="esm2_t30_150M_UR50D"
train_type="full_finetune"

python generate.py \
    --ckpt ${CKPT} \
    --test_set ${DATA_DIR}/test.json \
    --test_fasta ${DATA_DIR}/test_pred_${model_name}_${train_type}.fasta \
    --out ${CKPT_DIR}/results \
    --gpu 0 \
    --rabd_test \
    --rabd_sample 1 \
    --topk 100 \
    --mode ${MODE} | tee -a ${CKPT_DIR}/rabd_log.txt
