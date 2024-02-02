#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Ground truth directory
ground_truth_dir="${ROOT_DIR}/data/pdb_heavy/"

# Array of alphas for which we have generated data
alphas=(0 0.1 0.01 0.2 0.3 0.4 0.5 0.8 1 2 3 4 5)

for alpha in "${alphas[@]}"; do
    echo "Computing RMSD for alpha=${alpha}"

    # Construct the path based on the alpha value
    paths="cdrh3/cdr_design_esm2_650m_RL_alpha_${alpha}_cosim"
    generated_dir="${ROOT_DIR}/logs/fixedbb/${paths}/pdb/"
    output_file="${ROOT_DIR}/logs/fixedbb/${paths}/rmsd_results.txt"

    # Run the RMSD computation
    python ${ROOT_DIR}/compute_rmsd.py --ground_truth_dir ${ground_truth_dir} --generated_dir ${generated_dir} --output_file ${output_file}
done
