#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Array of CDR types and folds
cdr_types=(1 2 3)
folds=(0 1 2 3 4 5 6 7 8 9)

# Main loop for testing
for cdr in "${cdr_types[@]}"; do
    for fold in "${folds[@]}"; do
        
        # Define data directory dynamically
        DATA_DIR="${ROOT_DIR}/data/cdrh$cdr/fold_$fold/"

        # Process data
        python ${ROOT_DIR}/src/datamodules/datasets/antibody.py --data_dir=${DATA_DIR}
    done
done
