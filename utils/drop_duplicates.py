# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-09 10:24:45
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-22 16:22:35
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/drop_duplicates.py
Description: Drop duplicated and unlikely 
'''
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem

date = int(sys.argv[1])

working_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%d'%date
files = os.listdir(working_dir)
num = 0
for file in files:
    name = file.split('-')[0]
    if name == 'generated_smi':
        num += 1

frames = []
working_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%d'%date
for i in range(num):
    df = pd.read_csv(working_dir+'/' + 'generated_smi-%d.csv'%i, index_col=0)
    df.drop_duplicates(subset=['decoded_smiles'], keep='last',inplace=True)
    frames.append(df)

combined_df = pd.concat(frames)
combined_df.reset_index(drop=True,inplace=True)

combined_df.drop_duplicates(subset=['decoded_smiles'], keep='last',inplace=True)
combined_df.reset_index(drop=True,inplace=True)
print(combined_df)

# drop unlikely molecules
smiles = combined_df['decoded_smiles'].tolist()
nfails = 0
unlikely_indices = []
for idx, smi in enumerate(smiles):
    try:
        smi_cano = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical = True)
        if smi == smi_cano:
            pass
        else:
            unlikely_indices.append(idx)
    except:
        unlikely_indices.append(idx)
        nfails += 1
combined_df = combined_df.drop(unlikely_indices)
combined_df.reset_index(drop=True,inplace=True)
print('number of unlikely molecules =', nfails)
print('number of generated molecules =', combined_df.shape[0])

combined_df.to_csv(working_dir+'/' + 'generated_smi_all.csv')

