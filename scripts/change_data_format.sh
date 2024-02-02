#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Home directory (one step above Root directory)
HOME_DIR=$(cd "$(dirname "$0")"; cd ../..; pwd)
echo "Locate project at ${HOME_DIR}"

cdrs=(1 2 3)
folds=(0 1 2 3 4 5 6 7 8 9)
files=("test.json" "train.json" "valid.json")
output_files=("test_data.jsonl" "train_data.jsonl" "val_data.jsonl") # Output file names

# for i in "${!files[@]}"; do
#     DATA_DIR="${ROOT_DIR}/data/cdrh3/"
#     OUTPUT_DIR="${HOME_DIR}/RefineGNN/data/rabd/"
#     INPUT_FILE_PATH="${DATA_DIR}${files[i]}"
#     OUTPUT_FILE_PATH="${OUTPUT_DIR}${output_files[i]}"

#     echo "Processing $INPUT_FILE_PATH"
#     python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"
# done

for cdr in "${cdrs[@]}"; do
    for fold in "${folds[@]}"; do
        DATA_DIR="${ROOT_DIR}/data/cdrh${cdr}/fold_${fold}/"
        OUTPUT_DIR="${HOME_DIR}/RefineGNN/data/sabdab/hcdr${cdr}_cluster/fold_${fold}/"
        
        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        for i in "${!files[@]}"; do
            INPUT_FILE_PATH="${DATA_DIR}${files[i]}"
            OUTPUT_FILE_PATH="${OUTPUT_DIR}${output_files[i]}"

            if [[ -f "$INPUT_FILE_PATH" ]]; then
                echo "Processing $INPUT_FILE_PATH"
                python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"
            else
                echo "File not found: $INPUT_FILE_PATH"
            fi
        done
    done
done