# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-02 11:07:33
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-09 10:14:45
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/distribution.py
Description: 
'''

import os
import sys
import numpy as np
import pandas as pd

# ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_ls.csv',header=None).iloc[:,1:].to_numpy()
# de_smi_df = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_smi.csv')['decoded_smiles']
# fi_smi_df = pd.read_table('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/filtered/filtered.smi',header=None)
# new_npy = np.zeros((len(fi_smi_df),512))
# for i in range(len(de_smi_df)):
#     for j in range(len(fi_smi_df)):
#         if de_smi_df[i] == fi_smi_df[0][j]:
#             new_npy[j,:] = ls_npy[i,:]
# print(new_npy)



# datasets = ['DAT','SERT','NET','extended_hERG','filtered']
# # datasets= ['1']
# for dataset in datasets:
#     ls_npy = pd.read_csv(f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/{dataset}/ls-{dataset}.csv',header=None).to_numpy()
#     ls_npy = np.absolute(ls_npy)
#     # ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_ls.csv',header=None).iloc[:,1:].to_numpy()..
#     # ls_npy = np.absolute(new_npy)
#     print(ls_npy.shape)

#     distribution = np.sum(ls_npy,axis = 0)
#     # print(distribution)

#     import plotly.graph_objects as go
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=np.arange(0,512,1), y=distribution/ls_npy.shape[0]
#     ))

#     fig.update_layout(title='Styled Scatter',
#                   yaxis_zeroline=False, xaxis_zeroline=False)

#     import plotly.io as pio
#     pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/{dataset}_distribute.png', width=1921, height=1030) 
#     # pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/generated_distribute.png', width=1921, height=1030)


# # datasets = ['DAT','SERT','NET','extended_hERG','filtered']
# datasets= ['1']
# for dataset in datasets:
#     ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_1205.csv',header=None).to_numpy()
#     ls_npy = np.absolute(ls_npy)
#     # ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_ls.csv',header=None).iloc[:,1:].to_numpy()..
#     # ls_npy = np.absolute(new_npy)
#     print(ls_npy.shape)

#     distribution = np.sum(ls_npy,axis = 0)
#     # print(distribution)

#     import plotly.graph_objects as go
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=np.arange(0,512,1), y=distribution/ls_npy.shape[0]
#     ))

#     fig.update_layout(title='Styled Scatter',
#                   yaxis_zeroline=False, xaxis_zeroline=False)

#     import plotly.io as pio
#     pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/generator2_distribute.png', width=1921, height=1030) 
#     # pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/generated_distribute.png', width=1921, height=1030)


# datasets = ['DAT','SERT','NET','extended_hERG','filtered']
indices = list(range(820))
# indices.remove(25).remove(35).remove(36).remove(37).remove(63).remove(64).remove(82).remove(102).remove(103).remove(104).remove(112).remove(133).remove(149).remove(150)
total_dis = np.array(512)
total_num = 0
for i in indices:
    # ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated2/generated_ls_%d.csv'%i,header=None).to_numpy()
    # ls_npy = np.absolute(ls_npy)
    ls_npy = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generated_1207.csv',header=None).iloc[:,1:].to_numpy()
    ls_npy = np.absolute(ls_npy)
    print(ls_npy.shape)

    distribution = np.sum(ls_npy,axis = 0)
    # print(distribution)
    # total_dis += distribution
    # total_num += ls_npy.shape[0]

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(0,512,1), y=distribution/ls_npy.shape[0]
))

fig.update_layout(title='Styled Scatter',
            yaxis_zeroline=False, xaxis_zeroline=False)

import plotly.io as pio
# pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/filtered2_distribute.png', width=1921, height=1030) 
pio.write_image(fig, f'/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/generated2_distribute.png', width=1921, height=1030)
