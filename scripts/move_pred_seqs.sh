#!/bin/bash

# Base directories
src_base="/home/nayoung/ByProt/logs/fixedbb"
dest_base="/home/nayoung/MEAN/summaries"

# Declare an associative array to map source filenames to destination filenames
# declare -A rename_map
# rename_map=( ["test_iter1_denoise_tau0.fasta"]="test_pred_RL.fasta" ["train_iter1_denoise_tau0.fasta"]="train_pred_RL.fasta" ["valid_iter1_denoise_tau0.fasta"]="valid_pred_RL.fasta" )

model_names=("esm2_t6_8M_UR50D" "esm2_t12_35M_UR50D" "esm2_t30_150M_UR50D")
train_types=("train_from_scratch" "full_finetune")

for model_name in "${model_names[@]}"; do
    for train_type in "${train_types[@]}"; do
        # Declare and initialize the associative array inside the loop
        declare -A rename_map
        rename_map=( ["test_iter1_denoise_tau0.fasta"]="test_pred_${model_name}_${train_type}.fasta" ["train_iter1_denoise_tau0.fasta"]="train_pred_${model_name}_${train_type}.fasta" ["valid_iter1_denoise_tau0.fasta"]="valid_pred_${model_name}_${train_type}.fasta" )
        
        for cdrh in cdrh3; do
            # Source and destination directories
            src_dir="${src_base}/${cdrh}/${model_name}_${train_type}"
            dest_dir="${dest_base}/${cdrh}"
            
            # Make sure destination directory exists
            mkdir -p "${dest_dir}"

            # Copy and rename the files
            for src_file in "${!rename_map[@]}"; do
                dest_file="${rename_map[$src_file]}"
                if [[ -f "${src_dir}/${src_file}" ]]; then
                    cp "${src_dir}/${src_file}" "${dest_dir}/${dest_file}"
                fi
            done
        done
    done
done