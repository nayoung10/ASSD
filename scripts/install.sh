#!/bin/bash

# Define the project folder
PROJ_FOLDER=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${PROJ_FOLDER}"

# Create the conda environment
conda env create -f ${PROJ_FOLDER}/environment.yml

# Activate the conda environment
ENV_NAME=$(grep "name:" ${PROJ_FOLDER}/environment.yml | cut -d " " -f 2)
echo "Activating environment ${ENV_NAME}"
conda activate ${ENV_NAME}  

# Install pip packages
pip install -e ${PROJ_FOLDER}/vendor/esm
pip install -r ${PROJ_FOLDER}/requirements.txt