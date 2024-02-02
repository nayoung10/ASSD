#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Home directory (one step above Root directory)
HOME_DIR=$(cd "$(dirname "$0")"; cd ../..; pwd)
echo "Locate project at ${HOME_DIR}"

folds=(1 2 3 4 5 6 7 8 9 10)

# for i in "${!files[@]}"; do
#     DATA_DIR="${ROOT_DIR}/data/cdrh3/"
#     OUTPUT_DIR="${HOME_DIR}/RefineGNN/data/rabd/"
#     INPUT_FILE_PATH="${DATA_DIR}${files[i]}"
#     OUTPUT_FILE_PATH="${OUTPUT_DIR}${output_files[i]}"

#     echo "Processing $INPUT_FILE_PATH"
#     python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"
# done

for fold in "${folds[@]}"; do
    INPUT_FILE_PATH="${HOME_DIR}/ByProt_covid/summaries/docked_templates/test_${fold}/test_${fold}.json"
    OUTPUT_FILE_PATH="${HOME_DIR}/RefineGNN/data/docked_templates/test_${fold}/test_data_${fold}.jsonl"

    if [[ -f "$INPUT_FILE_PATH" ]]; then
        echo "Processing $INPUT_FILE_PATH"
        python "${ROOT_DIR}/change_data_format.py" --input_file_path "$INPUT_FILE_PATH" --output_file_path "$OUTPUT_FILE_PATH"
    else
        echo "File not found: $INPUT_FILE_PATH"
    fi
done
