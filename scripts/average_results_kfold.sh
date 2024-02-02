#!/bin/bash

ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/logs/fixedbb"
SUMMARY_FILE="summary_iter1_denoise_tau0.fasta"
CDR_TYPES="1,2,3"
FOLDS="0,1,2,3,4,5,6,7,8,9"

# alphas=(0.1)

# for alpha in "${alphas[@]}"; do
#     NAME="cdr_design_esm2_650m_RL_alpha_${alpha}_cosim"
#     python ${ROOT_DIR}/evaluation/average_results_kfold.py "$LOG_DIR" "$SUMMARY_FILE" "$CDR_TYPES" "$FOLDS" "$NAME"
# done

NAME="esm2_t6_8M_UR50D_full_finetune"
python ${ROOT_DIR}/evaluation/average_results_kfold.py "$LOG_DIR" "$SUMMARY_FILE" "$CDR_TYPES" "$FOLDS" "$NAME"