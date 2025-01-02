# Decoupled Sequence and Structure GEneration for Realistic Antibody Design

This is a public repository for our paper [Decoupled Sequence and Structure GEneration for Realistic Antibody Design](https://openreview.net/forum?id=CTkABQvnkm). 

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
git clone https://github.com/nayoung10/ASSD.git
cd ASSD

# setup environment
source scripts/install.sh
```

## Data Preparation

### Download 
1. Download ```all_structures.zip``` from the [SAbDab download page](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/). 
2. Move ```all_structures.zip``` to ```ASSD/MEAN```
3. Unzip with ```unzip all_structures.zip```

### Pre-processing for MEAN
We use the data-preprocessing scripts provided by [MEAN](https://github.com/THUNLP-MT/MEAN/).

```bash
cd MEAN # in ASSD/MEAN directory
bash scripts/prepare_data_kfold.sh summaries/sabdab_summary.tsv all_structures/imgt
bash scripts/prepare_data_rabd.sh summaries/rabd_summary.jsonl all_structures/imgt summaries/sabdab_all.json
bash scripts/prepare_data_skempi.sh summaries/skempi_v2_summary.jsonl all_structures/imgt summaries/sabdab_all.json
```

### Pre-processing for ESM2
```bash
# copy data splits to ASSD/data
mkdir -p ../data && \
rsync -avm --include='*/' \
--include='train.json' \
--include='valid.json' \
--include='test.json' \
--include='*.pdb' \
--exclude='*' \
summaries/ ../data/

# data pre-processing for ESM2
cd ../ # in ASSD directory
bash scripts/prepare_data_kfold.sh
bash scripts/prepare_data_rabd.sh
bash scripts/prepare_data_skempi.sh
```



## Benchmark Experiments

### Task 1: Sequence and Structure Modeling
We first fine-tune/evaluate the sequence design model [ESM2](https://www.pnas.org/doi/full/10.1073/pnas.2016239118), then use its sequence predictions \hat{s} to train/evaluate the structure prediction model [MEAN](https://arxiv.org/abs/2208.06073). For more information on running MEAN, please visit the [MEAN github page](https://github.com/THUNLP-MT/MEAN/). 


```bash
# Step 1: Sequence design
bash scripts/k_fold_train.sh # training sequence design model 
bash scripts/k_fold_eval.sh # evaluate AAR
bash scripts/average_results_kfold.sh # average results across all folds

# Step 2: Structure prediction 
bash scripts/generate_seqs_kfold.sh # generate sequence -- i.e., the input of the structure prediction model 
cd MEAN # in ASSD/MEAN directory 
GPU=0 bash scripts/k_fold_train.sh summaries 111 mean 9901
GPU=0 bash scripts/k_fold_eval.sh summaries 111 mean 0
```

### Task 2: Antibody-binding CDR-H3 Design

```bash
# Step 1: Sequence design
bash scripts/task2_train.sh # training sequence design model 
bash scripts/task2_eval.sh # evaluate AAR and CoSim

# Step 2: Structure prediction (MEAN)
bash scripts/generate_seqs_rabd.sh # generate sequence -- i.e., the input of the structure prediction model
cd MEAN # in ASSD/MEAN directory 
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash train.sh mean 3
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash rabd_test.sh 0
```

## Acknowledgements
We deeply appreciate the following works/repositories, on which our project heavily relies.
- [Structure-informed Language Models Are Protein Designers](https://github.com/BytedProtein/ByProt/tree/main) for training/testing pipelines of the sequence design model.
- [MEAN: Conditional Antibody Design as 3D Equivariant Graph Translation](https://github.com/THUNLP-MT/MEAN/) for data-preprocessing and the entire structure prediction model. 
- [Evolutionary Scale Modeling](https://github.com/facebookresearch/esm/tree/main) as the pLM adopted for the sequence design model. 
- [Scoring function for automated assessment of protein structure template quality](https://zhanggroup.org/TM-score/) for tm-scoring of the generated structures. 

## Citation
