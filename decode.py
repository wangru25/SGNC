# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-29 10:50:57
LastModifiedBy: Rui Wang
LastEditTime: 2023-08-01 15:18:27
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/decode.py
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

# # ===============================Decode Latent Space Feautures======================
# input_file = sys.argv[1]
# output_file = sys.argv[2]
# date = int(sys.argv[3])
# data_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_%d'%date #abs path
# os.system("python utils/LS_decode.py --input %s/%s --output %s/%s --use_gpu"%(data_dir, input_file, data_dir, output_file))


def get_subdirectories(path):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.startswith("generator_202")]
    dates = [folder.split('_')[1] for folder in folders]
    return dates

folder_dates = get_subdirectories('./results')
folder_dates.sort()
folder_dates = folder_dates[:-3]
print(folder_dates)

for i, date in enumerate(folder_dates):
    data_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated_2nd'
    save_dir = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/generated_2nd'
    input_file = 'generator_2nd_ls_%s.csv'%date
    output_file = 'generator_2nd-smi_%s.csv'%date
    os.system("python utils/LS_decode.py --input %s/%s --output %s/%s --use_gpu"%(data_dir, input_file, save_dir, output_file))
