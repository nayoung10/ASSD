#!/bin/bash
if [ -z ${GPU} ]; then
    GPU=0
fi
if [ -z ${MODE} ]; then
    MODE=111
fi
if [ -z ${DATA_DIR} ]; then
    DATA_DIR=/data/private/kxz/antibody/data/SAbDab
fi
if [ -z ${MODEL} ]; then
    MODEL=mcatt
fi
if [ -z ${CDR} ]; then
    CDR=3
fi
if [ -z ${RUN} ]; then
    RUN=5
fi

echo "Using GPU: ${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU

VERSION=0
if [ $1 ]; then
    VERSION=$1
fi
CKPT_DIR=${DATA_DIR}/ckpt/${MODEL}_CDR${CDR}_${MODE}/version_${VERSION}
CKPT=${CKPT_DIR}/checkpoint/best.ckpt

model_name=("esm2_t33_650M_UR50D")

echo "Using checkpoint: ${CKPT}"
python generate.py \
    --ckpt ${CKPT} \
    --test_set ${DATA_DIR}/test.json \
    --test_fasta ${DATA_DIR}/test_pred_${model_name}.fasta \
    --out ${CKPT_DIR}/results \
    --gpu 0 \
    --run ${RUN} \
    --mode ${MODE} | tee -a ${CKPT_DIR}/eval_log.txt
