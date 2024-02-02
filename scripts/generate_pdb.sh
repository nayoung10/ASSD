#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Array of paths and directories
alphas=(0 0.01 0.1 0.2 0.3 0.4 0.5 0.8 1 2 3 4 5)

for alpha in "${alphas[@]}"; do
    echo "INFO :: Processing alpha value: ${alpha}"  

    path="cdrh3/cdr_design_esm2_650m_RL_alpha_${alpha}_cosim"
    fasta_path="${ROOT_DIR}/logs/fixedbb/${path}/test_iter1_denoise_tau0.fasta"
    output_dir="${ROOT_DIR}/logs/fixedbb/${path}/pdb/"

    # Create the output directory if it doesn't exist
    mkdir -p "${output_dir}"

    # Run the Python script with the desired fasta_path and output_dir values
    python ${ROOT_DIR}/generate_pdb.py --fasta_path ${fasta_path} --output_dir ${output_dir}
done