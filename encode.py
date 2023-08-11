# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-05 14:03:01
LastModifiedBy: Rui Wang
LastEditTime: 2023-08-01 15:19:13
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/encode.py
Description: 
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from numpy.random import random
from numpy.random import randint
from torch.autograd import Variable  
from utils.predictor_network import *  

# ===============================Get Latent Space Feautures======================
input_file = sys.argv[1]
output_file = sys.argv[2]
date = int(sys.argv[3])
data_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%d'%date #abs path
os.system("python utils/run_LS.py -i %s/%s -o %s/%s --use_gpu --smiles_header decoded_smiles"%(data_dir, input_file, data_dir, output_file))

# def get_subdirectories(path):
#     folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.startswith("generator_202")]
#     dates = [folder.split('_')[1] for folder in folders]
#     return dates

# folder_dates = get_subdirectories('./results')
# folder_dates.sort()
# folder_dates = folder_dates[:-3]
# print(folder_dates)

# for i, date in enumerate(folder_dates):
#     data_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%s'%date #abs path
#     save_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated_2nd'
#     input_file = 'generated_smi_all.csv'
#     output_file = 'generator_2nd_ls_%s.csv'%date
#     os.system("python utils/run_LS.py -i %s/%s -o %s/%s --use_gpu --smiles_header decoded_smiles"%(data_dir, input_file, save_dir, output_file))
