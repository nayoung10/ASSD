#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Array of CDR types, folds, and data splits
cdr_types=(1 2 3)
# cdr_types=(1) next 
folds=(0 1 2 3 4 5 6 7 8 9)
data_splits=('train' 'valid')
alphas=(0.1)

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Main loop for testing
for alpha in "${alphas[@]}"; do
    for cdr in "${cdr_types[@]}"; do
        for fold in "${folds[@]}"; do
            for split in "${data_splits[@]}"; do
                
                # Define experiment name dynamically
                exp_name="fixedbb/cdrh$cdr/fold_$fold/cdr_design_esm2_650m_RL_alpha_${alpha}_cosim" # TODO: Fix this 

                # List of files that we want to back up and restore
                files_to_backup=("test_iter1_denoise_tau0.fasta" "native.fasta" "summary_iter1_denoise_tau0.fasta")

                # Check if the files exist and create backups if they do
                for file in "${files_to_backup[@]}"; do
                    if test -f "${ROOT_DIR}/logs/${exp_name}/$file"; then
                        cp "${ROOT_DIR}/logs/${exp_name}/$file" "${ROOT_DIR}/logs/${exp_name}/$file.bak"
                    fi
                done

                # Generate the fasta file 
                exp_path="${ROOT_DIR}/logs/${exp_name}"
                python ${ROOT_DIR}/test.py experiment_path=${exp_path} data_split=${split} ckpt_path=best.ckpt mode=predict task.generator.max_iter=1 task.generator.strategy='denoise'

                # Rename the main output file
                mv "${ROOT_DIR}/logs/${exp_name}/test_iter1_denoise_tau0.fasta" "${ROOT_DIR}/logs/${exp_name}/${split}_iter1_denoise_tau0.fasta"

                # Recover the backup files to their original names
                for file in "${files_to_backup[@]}"; do
                    if test -f "${ROOT_DIR}/logs/${exp_name}/$file.bak"; then
                        mv "${ROOT_DIR}/logs/${exp_name}/$file.bak" "${ROOT_DIR}/logs/${exp_name}/$file"
                    fi
                done
            done
        done
    done
done


