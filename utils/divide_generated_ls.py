# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-05 10:30:04
LastModifiedBy: Rui Wang
LastEditTime: 2023-01-05 09:40:30
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/divide_generated_ls.py
Description: 
'''
import os
import sys
import numpy as np
import pandas as pd

date = int(sys.argv[1])

def divide_list(in_df, n_strand):
    temp = np.arange(0,len(in_df),n_strand)
    if temp[-1] != len(in_df):
        out = np.append(temp, len(in_df))
    else:
        out = temp
    return out.tolist()

saving_path = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%d'%date
isExist = os.path.exists(saving_path)
if isExist is False:
    os.chdir('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results')
    os.system('mkdir generator_%d'%date)
    os.chdir('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder')

ls_df = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_%d.csv'%date, index_col=0)
strand = divide_list(ls_df,n_strand=4000)
print(ls_df.shape)
print(strand)

for i,val in enumerate(range(len(strand)-1)):
    low_idx = strand[i]
    upper_idx = strand[i+1]
    tmp_df = ls_df[low_idx:upper_idx]
    tmp_df.to_csv(saving_path + '/' + 'generated-%d.csv'%i, index=False, header=None)

