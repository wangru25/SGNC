# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-30 12:29:11
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-23 13:57:13
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/filtered.py
Description: 
'''
import os
import sys
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDConfig
sys.path.append('./')
import utils.SA_Score.sascorer as sascorer
from utils.predictor_network import * 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def read_dataset(feature_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file, header=None)
    X = df_X.values
    return X

# Training settings
parser = argparse.ArgumentParser(description='filter')    
parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='latent_size')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to generate (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--device',default="1", type=str,
                        help="number of cuda visible devise")
parser.add_argument('--seed', type=int, default=1209, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--date',default="20221209", type=int,
                        help="date")
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.device
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# predictor_network
fg_len = args.latent_size
dropout = 0.0
network_layers=[512,1024,512]

if(len(network_layers)==1):
    DAT_predictor_model = Net1(fg_len,network_layers[0],dropout).to(device)
elif(len(network_layers)==2):
    DAT_predictor_model = Net2(fg_len,network_layers[0],network_layers[1],dropout).to(device)
elif(len(network_layers)==3):
    DAT_predictor_model = Net3(fg_len,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
    NET_predictor_model = Net3(fg_len,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
    SERT_predictor_model = Net3(fg_len,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
    hERG_predictor_model = Net3(fg_len,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
elif(len(network_layers)==4):
    DAT_predictor_model = Net4(fg_len,network_layers[0],network_layers[1],network_layers[2],network_layers[3],dropout).to(device)
elif(len(network_layers)==5):
    DAT_predictor_model = Net5(fg_len,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4],dropout).to(device)
elif(len(network_layers)==6):
    DAT_predictor_model = Net6(fg_len,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4], network_layers[5], dropout).to(device)

#load predictor model
DAT_predictor_model.load_state_dict(torch.load('./model/DAT_predictor_model.pt'))
DAT_predictor_model.eval()

NET_predictor_model.load_state_dict(torch.load('./model/NET_predictor_model.pt'))
NET_predictor_model.eval()

SERT_predictor_model.load_state_dict(torch.load('./model/SERT_predictor_model.pt'))
SERT_predictor_model.eval()

hERG_predictor_model.load_state_dict(torch.load('./model/extended_hERG_predictor_model.pt'))
hERG_predictor_model.eval()


write_file = open('./results/generator_%d/0_selected_outer_predictor.csv'%args.date,'w')
write_file.write('smiles,DAT_BA,SERT_BA,NET_BA,hERG_BA\n')
write_file.close()
write_file = open('./results/generator_%d/0_selected_outer_predictor.csv'%args.date,'a+')
write_smi = open('./results/generator_%d/0_selected_outer_predictor.smi'%args.date,'w')
# x_test = read_dataset('../results/generator_%d/generated_ls_all.csv')  
# # x_test = pd.read_csv('./results/generated_ls.csv',header=None).iloc[:,1:].to_numpy()
# smi_df = pd.read_table('./data/filtered/filtered.smi', header=None)


# x_test = read_dataset('./data/generated/generated_ls_%d.csv')
# smi_df = pd.read_csv('./data/generated/generated_smi_%d.csv'%idx, index_col=0)
x_test = read_dataset('./results/generator_%d/generated_ls_all.csv'%args.date)
smi_df = pd.read_csv('./results/generator_%d/generated_smi_all.csv'%args.date, index_col=0)
vector = torch.from_numpy(x_test).float().to(device)
DAT_ba = DAT_predictor_model(vector)
NET_ba = NET_predictor_model(vector)
SERT_ba = SERT_predictor_model(vector)
hERG_ba = hERG_predictor_model(vector)

num = 0
for i in range(x_test.shape[0]):
    # if DAT_ba[i] < - 9.54:
    # if NET_ba[i] < - 9.54:
    # if SERT_ba[i] < - 9.54:
    # print(DAT_ba[i],NET_ba[i],SERT_ba[i],hERG_ba[i])
    if DAT_ba[i] < - 9.54 and SERT_ba[i] < -9.54 and NET_ba[i] < - 9.54 and hERG_ba[i] > - 8.18:
        num += 1
    # if DAT_ba[i] < - 9.54 and NET_ba[i] < - 9.54 and SERT_ba[i] < - 9.54 and hERG_ba[i] > -8.18: 
        # write_file.write(f'{smi_df[0][i]},{DAT_ba[i]},{SERT_ba[i]},{NET_ba[i]},{hERG_ba[i]}\n')
        write_file.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(smi_df['decoded_smiles'][i],DAT_ba[i],SERT_ba[i],NET_ba[i],hERG_ba[i]))
        write_smi.write('%s\n'%smi_df['decoded_smiles'][i])
print('number of molecules that pass the binding test = ', num)
print('=================================================================')

write_file.close()
write_smi.close()