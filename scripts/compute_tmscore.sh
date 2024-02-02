#!/bin/bash

# Set visible devices
export CUDA_VISIBLE_DEVICES=4,5,6
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Ground truth directory
ground_truth_dir="${ROOT_DIR}/data/pdb_heavy/"

# Path to TMscore executable
tmscore_exe_path="/home/nayoung/TMscore"

# Array of alphas for which we have generated data
alphas=(0)
# alphas=(0 0.1 0.01 0.2 0.5 1 2 3 4 5)

for alpha in "${alphas[@]}"; do
    echo "Computing TM-score for alpha=${alpha}"

    # Construct the path based on the alpha value
    paths="cdrh3/cdr_design_esm2_650m_RL_alpha_${alpha}"
    generated_dir="${ROOT_DIR}/logs/fixedbb/${paths}/pdb/"
    output_file="${ROOT_DIR}/logs/fixedbb/${paths}/tmscore_results.txt"

    # Run the TM-score computation
    python ${ROOT_DIR}/compute_tmscore.py --ground_truth_dir ${ground_truth_dir} --generated_dir ${generated_dir} --output_file ${output_file} --tmscore_exe_path ${tmscore_exe_path}

done
