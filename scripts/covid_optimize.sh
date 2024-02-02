#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=7
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export PROJECT_ROOT="${ROOT_DIR}"

SAVE_DIR="${ROOT_DIR}/logs/covid/results/"
mkdir -p "$SAVE_DIR"

python "${ROOT_DIR}/covid_optimize.py" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 \
    --epochs 100000 \