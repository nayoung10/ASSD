#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Home directory (one step above Root directory)
HOME_DIR=$(cd "$(dirname "$0")"; cd ../..; pwd)
echo "Locate project at ${HOME_DIR}"

# files=("train.json" "valid.json")
# output_files=("train_data.jsonl" "val_data.jsonl") # Output file names

# # pre-training files 
# for i in "${!files[@]}"; do
#     DATA_DIR="${ROOT_DIR}/data/"
#     OUTPUT_DIR="${HOME_DIR}/RefineGNN/data/SKEMPI/"

#     # Create output directory if it doesn't exist
#     mkdir -p "$OUTPUT_DIR"

#     INPUT_FILE_PATH="${DATA_DIR}${files[i]}"
#     OUTPUT_FILE_PATH="${OUTPUT_DIR}${output_files[i]}"

#     echo "Processing $INPUT_FILE_PATH"
#     python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"
# done

# ita files
INPUT_FILE_PATH="${ROOT_DIR}/data/skempi_all.json"
OUTPUT_DIR="${HOME_DIR}/ByProt_covid/data/SKEMPI/"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE_PATH="${OUTPUT_DIR}skempi_all.jsonl"

echo "Processing $INPUT_FILE_PATH"
python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"