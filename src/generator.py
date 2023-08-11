# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-05 00:37:16
LastModifiedBy: Rui Wang
LastEditTime: 2023-07-14 16:44:25
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/generator.py
Description: 
'''
import os
import sys
sys.path.append('./')
import pickle
import argparse
import numpy as np
import pandas as pd
from numpy.random import random
from numpy.random import randint
from scipy.integrate import solve_ivp
from utils.predictor_network import *  



# Training settings
parser = argparse.ArgumentParser(description='GNC Generator')    
dropout=0.0 # default=0.3   
parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='latent_size')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N',
                        help='number of epochs to generate (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--device',default="1", type=str,
                        help="number of cuda visible devise")
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--generator-with-reference', action='store_true', default=True,
                        help='generator_with_reference')   
parser.add_argument('--seed-start-point', type=float, default=7, metavar='M',
                        help='output binding affinity start point (default: -10.0)') 
parser.add_argument('--seed-stop-point', type=float, default=0, metavar='M',
                        help='output binding affinity stop point (default: -14.0)')
parser.add_argument('--output-upper-point', type=float, default=-8.18, metavar='M',
                        help='output binding affinity start point (default: -10.0)') 
parser.add_argument('--output-lower-point', type=float, default=-9.54, metavar='M',
                        help='output binding affinity stop point (default: -14.0)')
parser.add_argument('--save-optimized-vector-path', type=str,
                        help='Save path optimized-vector')
parser.add_argument('--save-optimized-binding-affinity', type=str,
                        help='Save path optimized binding affinity')
parser.add_argument('--init-vec', default="Provided", type=str,
                        help='Provided or Random')  
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.device

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#functions
def read_latent_space(feature_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file,header=None)
    X = df_X.values # convert values in dataframe to numpy array (features)
    return X

class generator_net2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(generator_net2, self).__init__()
        self.input_dim=input_dim
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, 2*self.input_dim)
        self.fc2 = nn.Linear(2*self.input_dim, 2*self.input_dim)
        self.fc3 = nn.Linear(2*self.input_dim, self.input_dim)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.drop(x)
        x = torch.tanh(self.fc2(x))
        x = self.drop(x)
        x = torch.tanh(self.fc3(x))
        return x

def Tanimoto_coefficient(l1,l2):
    product=torch.mul(l1,l2)
    sum_product=torch.sum(product)
    l1_squared=torch.mul(l1,l1)
    sum_l1_squared=torch.sum(l1_squared)
    l2_squared=torch.mul(l2,l2)
    sum_l2_squared=torch.sum(l2_squared)
    tc=torch.div(sum_product,(sum_l1_squared+sum_l2_squared-sum_product))
    return(tc)

class binding_affinity_and_sim_reference_loss(nn.Module):
    def __init__(self):
        super(binding_affinity_and_sim_reference_loss, self).__init__()

    def forward(self, data, ref, target_sim):       
        loss = torch.abs(Tanimoto_coefficient(data,ref)-target_sim)
        # return torch.tensor(loss, requires_grad=True)
        return loss.clone().detach().requires_grad_(True)

def generator_vector(args, alpha, ref_coefficients, refs, noise, init_vec, epoch):
    '''
    -------
    Input: 
        - init_vec: init vector with shape (512,): selected latent space vector from DAT, SERT, NET with relatively high similarity. npy
        - ref_coffecients: a list of coffecients with sum = 1. Here we have 3 coffecients (for DAT, SERT, NET) npy shape (3,1)
        - refs: a list of reference vectors. Here we have 3 refs (from DAT, SERT, NET) npy shape (3,512)
        - alpha: a small hyperp-parameters
        - noise: random noise (512,) npy
    -------
    Output:
        - outputs: (512, N_times). tensor
    '''
    F = lambda t, X: -alpha * X + np.sum(alpha * ref_coefficients * refs, axis=0) + noise
    t_eval = np.arange(0, epoch+500, 500)
    sol = solve_ivp(F, [0, epoch], init_vec, t_eval=t_eval)
    outputs = sol.y
    outputs = torch.from_numpy(outputs).float().to(device)
    return outputs

def main():
    #load latent space and label
    alpha = 0.15
    noise = np.array([random()*0.2-0.1 for i in range(512)])
    # noise = np.zeros(512)
    refs = read_latent_space('./data/reference/ls-reference.csv')
    refs_tensor = torch.from_numpy(refs).to(device)
    init_vec_ls = read_latent_space('./data/init_vec/ls-init_vec.csv')
    # init_vec_ls = read_latent_space('./data/init_vec/ls-init_vec_candidate_3.csv')
    ref_coefficients = np.array([[0.35,0.35,0.3]]).reshape(-1,1)
    if args.init_vec == 'Provided':
        seed_id = randint(0,len(init_vec_ls))
        init_vec = init_vec_ls[seed_id].ravel()
    
    # predictor_network
    network_layers=[512,1024,512]
    fg_len=args.latent_size
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

    # load predictor model
    DAT_predictor_model.load_state_dict(torch.load('./model/DAT_predictor_model.pt'))
    DAT_predictor_model.eval()

    NET_predictor_model.load_state_dict(torch.load('./model/NET_predictor_model.pt'))
    NET_predictor_model.eval()

    SERT_predictor_model.load_state_dict(torch.load('./model/SERT_predictor_model.pt'))
    SERT_predictor_model.eval()
    
    hERG_predictor_model.load_state_dict(torch.load('./model/extended_hERG_predictor_model.pt'))
    hERG_predictor_model.eval()

    write_file = open(str(args.save_optimized_binding_affinity),"a+")
    # write_file.write('DAT_BA,NET_BA,SERT_BA,hERG_BA,DAT_Sim,NET_Sim,SERT_Sim\n')
            
    vector_lists=[]
    vectors_tensor = generator_vector(args, alpha, ref_coefficients, refs, noise, init_vec, args.epochs)
    for i in range(vectors_tensor.shape[1]):
        vector = vectors_tensor[:,i]
        DAT_ba = DAT_predictor_model(vector)
        NET_ba = NET_predictor_model(vector)
        SERT_ba = SERT_predictor_model(vector)
        hERG_ba = hERG_predictor_model(vector)
        # print(DAT_ba.detach().cpu().numpy(),SERT_ba.detach().cpu().numpy(),NET_ba.detach().cpu().numpy(),hERG_ba.detach().cpu().numpy())
        # # print(vector)
        # new_vector = vector.detach().cpu().numpy().reshape(1, -1)
        # new_vector.fillna(new_vector.mean())
        # hERG_ba = reg.predict(new_vector)

        # if(DAT_ba <= args.output_lower_point and NET_ba <= args.output_lower_point and SERT_ba <= args.output_lower_point and hERG_ba >= args.output_upper_point):
        if(DAT_ba <= args.output_lower_point and NET_ba <= args.output_lower_point and SERT_ba <= args.output_lower_point):
            print(DAT_ba.detach().cpu().numpy(),SERT_ba.detach().cpu().numpy(),NET_ba.detach().cpu().numpy(),hERG_ba.detach().cpu().numpy())
            vector_list = torch.reshape(vector,(-1,)) 
            vector_list = vector_list.tolist()
            vector_lists.append(vector_list)                
            # print(vector_list, file=fo)                    
            DAT_sim = Tanimoto_coefficient(vector,refs_tensor[0])
            SERT_sim = Tanimoto_coefficient(vector,refs_tensor[1])
            NET_sim = Tanimoto_coefficient(vector,refs_tensor[2])
            # write_file.write(f'{DAT_ba.float():.4f},{SERT_ba.float():.4f},{NET_ba.float():.4f},{hERG_ba.float():.4f},{DAT_sim:.4f},{SERT_sim:.4f},{NET_sim:.4f}\n')
            write_file.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(DAT_ba.float(),SERT_ba.float(),NET_ba.float(),hERG_ba.float(),DAT_sim,SERT_sim,NET_sim))
    df = pd.DataFrame(vector_lists)
    df.to_csv(str(args.save_optimized_vector_path), mode='a+', header=False)
        
if __name__ == '__main__':
    for i in range(30):
        main()
