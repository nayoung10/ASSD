#!/bin/bash

# Base directories
src_base="/home/nayoung/ByProt/logs/fixedbb"
dest_base="/home/nayoung/MEAN/summaries"

# Declare an associative array to map source filenames to destination filenames
# declare -A rename_map
# rename_map=( ["test_iter1_denoise_tau0.fasta"]="test_pred_RL.fasta" ["train_iter1_denoise_tau0.fasta"]="train_pred_RL.fasta" ["valid_iter1_denoise_tau0.fasta"]="valid_pred_RL.fasta" )

alphas=(0.1)

for alpha in "${alphas[@]}"; do
    # Declare and initialize the associative array inside the loop
    declare -A rename_map
    rename_map=( ["test_iter1_denoise_tau0.fasta"]="test_pred_RL_alpha_${alpha}.fasta" ["train_iter1_denoise_tau0.fasta"]="train_pred_RL_alpha_${alpha}.fasta" ["valid_iter1_denoise_tau0.fasta"]="valid_pred_RL_alpha_${alpha}.fasta" )
    
    for cdrh in cdrh1 cdrh2 cdrh3; do
        # Iterate over fold values
        for fold in {0..9}; do
            # Source and destination directories
            src_dir="${src_base}/${cdrh}/fold_${fold}/cdr_design_esm2_650m_RL_alpha_${alpha}_cosim"
            dest_dir="${dest_base}/${cdrh}/fold_${fold}"
            
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