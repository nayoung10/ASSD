#!/bin/bash

# Define the project folder
PROJ_FOLDER=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${PROJ_FOLDER}"

# Create the conda environment
conda env create -f environment.yml

# Activate the conda environment
ENV_NAME=$(grep "name:" environment.yml | cut -d " " -f 2)
echo "Activating environment ${ENV_NAME}"
conda activate ${ENV_NAME}  

# Install pip packages
pip install -e vendor/esm
pip install -r requirements.txt
pip install -e .

# TMscore
EVAL_FOLDER=MEAN/evaluation
g++ -static -O3 -ffast-math -lm -o ${EVAL_FOLDER}/TMscore ${EVAL_FOLDER}/TMscore.cpp