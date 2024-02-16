#!/bin/bash

# Root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"


DATA_DIR="${ROOT_DIR}/data/cdrh3/"

# Process data
python ${ROOT_DIR}/src/datamodules/datasets/antibody.py --data_dir=${DATA_DIR}
