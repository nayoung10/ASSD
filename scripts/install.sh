#!/bin/bash

# Define the project folder
PROJ_FOLDER=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${PROJ_FOLDER}"

# Create the conda environment
conda env create -f environment.yml

ENV_NAME=$(grep "name:" environment.yml | cut -d " " -f 2)

if conda env list | grep -q "^${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} exists. Proceeding with package installations."

    conda run -n ${ENV_NAME} pip install -e vendor/esm
    conda run -n ${ENV_NAME} pip install -r requirements.txt
    conda run -n ${ENV_NAME} pip install -e .

    EVAL_FOLDER=MEAN/evaluation
    g++ -static -O3 -ffast-math -lm -o ${EVAL_FOLDER}/TMscore ${EVAL_FOLDER}/TMscore.cpp

    # Activate the environment
    echo "Activating environment ${ENV_NAME}"
    conda activate ${ENV_NAME}

else
    echo "INFO:: Installation failed. Environment ${ENV_NAME} not found. Please check the environment.yml file."
fi