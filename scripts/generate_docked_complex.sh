#!/bin/bash

# Set up the root directory
ROOT_DIR=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "Locate project at ${ROOT_DIR}"

# Directory where PDB files are located
PDB_SPLIT_DIR="${ROOT_DIR}/data/pdb_split"
OUTPUT_DIR="${ROOT_DIR}/data/lightdock"

# Number of swarms and glowworms
SWARMS=1
STEPS=100
GEN_NUM=10

# Loop through each antibody PDB file in the pdb_split directory
for ANTIBODY_PDB in "${PDB_SPLIT_DIR}"/*_antibody.pdb; do
    # Corresponding antigen file
    PDB_ID=$(basename "${ANTIBODY_PDB}" "_antibody.pdb")
    ANTIGEN_PDB="${PDB_SPLIT_DIR}/${PDB_ID}_antigen.pdb"

    # Check if the antigen file exists
    if [[ ! -f "${ANTIGEN_PDB}" ]]; then
        echo "Antigen file for ${PDB_ID} not found, skipping..."
        continue
    fi

    # Create a directory for this PDB ID
    WORKING_DIR="${OUTPUT_DIR}/${PDB_ID}"
    mkdir -p "${WORKING_DIR}"
    cd "${WORKING_DIR}"

    # Step 1: Setup LightDock
    echo "Setting up LightDock for ${PDB_ID}..."
    lightdock3_setup.py "${ANTIBODY_PDB}" "${ANTIGEN_PDB}" -s ${SWARMS} --noxt --noh --now -anm

    # Step 2: Run LightDock
    echo "Running LightDock for ${PDB_ID}..."
    lightdock3.py setup.json ${STEPS} -c 1 -l 0

    # Step 3: Generate conformations for the best swarm
    echo "Generating conformations for swarm_0 of ${PDB_ID}..."
    cd swarm_0
    lgd_generate_conformations.py "${ANTIBODY_PDB}" "${ANTIGEN_PDB}" gso_${STEPS}.out ${GEN_NUM}

    echo "LightDock process completed for ${PDB_ID}."
    cd "${PDB_SPLIT_DIR}"  # Go back to the PDB split directory for the next iteration
done

echo "All LightDock processes completed."
