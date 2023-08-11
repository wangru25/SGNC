# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-02 10:55:31
LastModifiedBy: Rui Wang
LastEditTime: 2023-08-01 16:38:22
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/recovery.py
Description: 
'''
import os
import sys
import numpy as np
import pandas as pd
from preprocessing import preprocess_smiles
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# # For 4 datasets
# datasets = ['DAT','NET','SERT','hERG']
# for dataset in datasets:
#     smi_df_1 = pd.read_table('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/%s/%s.smi'%(dataset,dataset),header=None)
#     smi_df_2 = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/%s/decoded-%s.csv'%(dataset,dataset))
#     num = 0
#     for i in range(len(smi_df_1)):
#         # print(i)
#         smi_1 = smi_df_1[0][i]
#         smi_2 = smi_df_2['decoded_smiles'][i]

#         try:
#             new_smi_1 = preprocess_smiles(smi_1)
#             new_smi_2 = preprocess_smiles(smi_2)
#             if new_smi_1 == new_smi_2:
#                 pass
#             else:
#                 num += 1
#         except:
#             num += 1
#     print("reconstruction rate of %s is: "%dataset, (len(smi_df_1)-num)/len(smi_df_1))

# For 2nd generations
def get_subdirectories(path):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.startswith("generator_202")]
    dates = [folder.split('_')[1] for folder in folders]
    return dates

folder_dates = get_subdirectories('../results')
folder_dates.sort()
folder_dates = folder_dates[:-3]
print(folder_dates)


smi_df_1 = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated_2nd/generated_2nd.csv')
smi_df_2 = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated_2nd/generator_2nd-smi.csv')
num = 0
for i in range(len(smi_df_1)):
    smi_1 = smi_df_1['decoded_smiles'][i]
    smi_2 = smi_df_2['decoded_smiles'][i]

    try:
        new_smi_1 = preprocess_smiles(smi_1)
        new_smi_2 = preprocess_smiles(smi_2)
        if new_smi_1 == new_smi_2:
            pass
        else:
            num += 1
    except:
        num += 1
print("reconstruction rate of is: ", (len(smi_df_1)-num)/len(smi_df_1))
