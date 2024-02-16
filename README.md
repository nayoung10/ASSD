# Anfinsen Goes Neural (AGN): a Graphical Model for Conditional Antibody Design

This is a public repository for our paper [Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design](https://arxiv.org/abs/2402.05982). 

## Overview

- [Environment](#environmental-setup)
    - [Conda](#conda)
- [Data Preparation](#data-preparation)
- [Benchmark Experiments](#benchmark-experiments)
    - [Task 1: Sequence and Structure Modeling](#task-1-sequence-and-structure-modeling)
    - [Task 2: Antibody-binding CDR-H3 Design](#task-2-antibody-binding-cdr-h3-design)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Environmental Setup
### Conda

```bash
# clone project
git clone https://github.com/lkny123/AGN.git
cd AGN

# setup environment
bash scripts/install.sh
```

## Data Preparation

### Download 
1. Download ```all_structures.zip``` from the [SAbDab download page](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/). 
2. Move ```all_structures.zip``` to ```AGN/MEAN```
3. Unzip with ```unzip all_structures.zip```

### Pre-processing for MEAN
We use the data-preprocessing scripts provided by [MEAN](https://github.com/THUNLP-MT/MEAN/).

```bash
cd MEAN # in AGN/MEAN directory
bash scripts/prepare_data_kfold.sh summaries/sabdab_summary.tsv all_structures/imgt
bash scripts/prepare_data_rabd.sh summaries/rabd_summary.jsonl all_structures/imgt summaries/sabdab_all.json
bash scripts/prepare_data_skempi.sh summaries/skempi_v2_summary.jsonl all_structures/imgt summaries/sabdab_all.json
```

### Pre-processing for ESM2
```bash
# copy data splits to AGN/data
mkdir -p ../data && \
rsync -avm --include='*/' \
--include='train.json' \
--include='valid.json' \
--include='test.json' \
--include='*.pdb' \
--exclude='*' \
summaries/ ../data/

# data pre-processing for ESM2
cd ../ # in AGN directory
bash scripts/prepare_data_kfold.sh
bash scripts/prepare_data_rabd.sh
bash scripts/prepare_data_skempi.sh
```



## Benchmark Experiments

### Task 1: Sequence and Structure Modeling
We first fine-tune/evaluate the sequence design model [ESM2](https://github.com/BytedProtein/ByProt/tree/main), then use its sequence predictions \hat{s} to train/evaluate the structure prediction model [MEAN](https://github.com/THUNLP-MT/MEAN/). 


```bash
# Step 1: Sequence design
bash scripts/k_fold_train.sh # training sequence design model 
bash scripts/k_fold_eval.sh # evaluate AAR

# Step 2: Structure prediction 
bash scripts/generate_seqs_kfold.sh # generate sequence -- i.e., the input of the structure prediction model 
bash MEAN/scripts/prepare_data_kfold.sh summaries/sabdab_summary.tsv all_structures/imgt
GPU=0 bash MEAN/scripts/k_fold_train.sh summaries 111 mean 9901
GPU=0 bash MEAN/scripts/k_fold_eval.sh summaries 111 mean 0
```

### Task 2: Antibody-binding CDR-H3 Design

```bash
# Step 1: Sequence design
bash scripts/task2_train.sh # training sequence design model 
bash scripts/task2_eval.sh # evaluate AAR and CoSim

# Step 2: Structure prediction (MEAN)
bash MEAN/scripts/generate_seqs.sh # generate sequence -- i.e., the input of the structure prediction model
bash MEAN/scripts/prepare_data_rabd.sh summaries/rabd_summary.jsonl all_structures/imgt summaries/sabdab_all.json
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash MEAN/train.sh mean 3
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash MEAN/rabd_test.sh 0
```

## Acknowledgements
We deeply appreciate the following works/repositories, on which our project heavily relies.
- [Structure-informed Language Models Are Protein Designers](https://github.com/BytedProtein/ByProt/tree/main)
- [MEAN: Conditional Antibody Design as 3D Equivariant Graph Translation](https://github.com/THUNLP-MT/MEAN/)
- [Evolutionary Scale Modeling](https://github.com/facebookresearch/esm/tree/main)

## Citation
