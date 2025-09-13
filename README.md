# Stochastic-based Generative Network Complex (SGNC)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python package for the Stochastic-based Generative Network Complex, designed for drug discovery and molecular generation using advanced machine learning techniques.


## Overview

The Stochastic-based Generative Network Complex (SGNC) is a comprehensive framework for molecular generation and drug discovery. It combines predictive models, generative networks, and filtering mechanisms to produce novel drug-like molecules with desired properties.

## Workflow

The SGNC workflow consists of five main stages:

![SGNC Workflow](img/SGNCWorkflow.pdf)


## Installation

### Prerequisites

- Python 3.6 or higher
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SGNC
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate sgnc
```

3. Install additional dependencies (if needed):
```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate molecular candidates
python generator.py --date 20221209

# 2. Apply ADMET and SA filters
python src/filtered.py --date 20221209

# 3. Check reproduction rate
python src/properties.py 20221209
```

## Detailed Usage

### 1. Predictive Model Preparation

#### 1.1 Training Datasets

The framework includes pre-trained models for multiple targets:
- **DAT** (Dopamine Transporter)
- **NET** (Norepinephrine Transporter) 
- **SERT** (Serotonin Transporter)
- **hERG** (Human Ether-a-go-go Related Gene)

Training data is located in the `data/` directory with corresponding SMILES files and labels.

#### 1.2 Train Predictive Model

```bash
# Train a new predictive model
python src/predictor_ANN.py --target DAT --epochs 100
```

A python package for the Stochastic-based Generative Network Complex

#### 1 Preperation for Predictive Models

##### 1.1 Training datasets 

##### 1.2 Train predictive model

#### 2 Preperation for Reference and Init Vector

##### 2.1 Reference 

##### 2.2 Init vector

#### 3 Generator Model

1. Submit thousands of jobs to generate latent space vectors.

   ```bash
   cd sbatch
   python submit_generator.py 20221209
   ```

2. Divide generated latent space vectors to sub-files. Each sub-file has 2000 records.

   ```bash
   cd ..
   python ./utils/divide_generated_ls.py 20221209
   ```

3. Decode all generated latent space vectors to smiles.

   ```bash
   cd sbatch
   python submit_decode.py 20221209
   ```

4. Drop duplicated and unlikely smiles. 

   ```bash
   cd ..
   python ./utils/drop_duplicates.py 20221209
   ```

#### 4 Filtered Model

1. Encode generated smiles to latent space vectors

   ```bash
   cd sbatch
   python submit_encoder.py 20221209
   ```

2. Binding affinity test

   ```bash
   cd ..
   python ./src/filtered.py --date 20221209
   ```

3. ADMET and SAS test

   Test ADMET on a online server: [ADMET](https://admetmesh.scbdd.com/service/screening/cal) and download a csv file. Then transfer this file to server

   ```bash
   scp ADMET.csv wangru25@hpcc.msu.edu:/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_20221209
   ```

   Then check if there is a molecule that falls in the optimal range. 

## Contact

- **Author**: Rui Wang
- **Email**: wangru25@msu.edu
- **Institution**: Michigan State University

### Reference
[1] Wang, R., Feng, H. and Wei, G.W., 2023. ChatGPT in Drug Discovery: A Case Study on Anticocaine Addiction Drug Development with Chatbots. Journal of Chemical Information and Modeling, 63(22), pp.7189-7209.
