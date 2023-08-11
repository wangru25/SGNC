# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-08 15:08:31
LastModifiedBy: Rui Wang
LastEditTime: 2022-11-28 01:20:06
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
from utils.predictor_network import *  



# Training settings
parser = argparse.ArgumentParser(description='GNC Generator')    
dropout=0.0 # default=0.3   
dataset = 'DAT' 
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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--cycle', type=int, default=200, metavar='N',
                        help='number of cycles for each seed') 
parser.add_argument('--generator-with-reference', action='store_true', default=True,
                        help='generator_with_reference')   
parser.add_argument('--target-sim', type=float, metavar='M',  
                        help='target similarity value') 
parser.add_argument('--target1-ba', type=float, metavar='M', 
                        help='target1 binding affinity') 
parser.add_argument('--target2-bbb', type=float, metavar='M', 
                        help='target2 blood-brain barrier') 
parser.add_argument('--sim-k', type=float, default=10, metavar='M',
                        help='Constant for similarity') 
parser.add_argument('--bbb-k', type=float, default=1.0, metavar='M',
                        help='Constant for blood-brain barrier') 
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

def predictor_test(args, predictor_model, device, data):
#    model.eval()
    data = data.to(device)
    output = predictor_model(data)
#    output_list=torch.reshape(output,(-1,))            
    return(output)
        
def binding_affinity_predictor(data, predictor_model):    
    return(predictor_test(args, predictor_model, device, data))
        
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

class binding_affinity_loss(nn.Module):
    def __init__(self):
        super(binding_affinity_loss, self).__init__()

    def forward(self, data, predictor_model):
        ba=binding_affinity_predictor(data, predictor_model)
        # return torch.tensor(ba, requires_grad=True)
        return ba.clone().detach().requires_grad_(True)

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
    
def generator_vector(args, ref, data, alpha_list, beta, target_sim): #low energy generating
    # data is gaussian noise
    # ref 
    data = data.to(device)
    ref = torch.from_numpy(ref).to(device)
    output = data
    sref = len(ref) 
    temp = torch.zeros_like(data)
    for i in range(sref):
        # print(ref[i].shape)
        # print(data.shape)
        temp += beta*(alpha_list[i]*(ref[i] - data))
    output += temp + data 
    sim_loss = binding_affinity_and_sim_reference_loss()
    loss = sim_loss(output, ref, target_sim)   
    return output, loss

def main():
    #load latent space and label
    ref_ls=read_latent_space('./data/reference/ls-reference.csv')
    # init_vec_ls = read_latent_space('./data/init_vec/ls-init_vec.csv')
    sref=len(ref_ls)
    
    # load seed and reference
    # target1_ba=float(args.target1_ba)
    # target2_bbb=float(args.target2_bbb)
    # bbb_k=args.bbb_k
    
    # predictor_network
    network_layers=[512,1024,512]

    # predictor network structure
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

#load predictor model
    DAT_predictor_model.load_state_dict(torch.load('./model/DAT_predictor_model.pt'))
    DAT_predictor_model.eval()

    NET_predictor_model.load_state_dict(torch.load('./model/NET_predictor_model.pt'))
    NET_predictor_model.eval()

    SERT_predictor_model.load_state_dict(torch.load('./model/SERT_predictor_model.pt'))
    SERT_predictor_model.eval()
    
    hERG_predictor_model.load_state_dict(torch.load('./model/extended_hERG_predictor_model.pt'))
    hERG_predictor_model.eval()

        
#generate low energy
    sim_k = args.sim_k
    target_sim = float(args.target_sim)
        
    # fo=open(args.save_optimized_vector_path,"w")
    write_file = open(str(args.save_optimized_binding_affinity),"a+")
            
    vector_lists=[]    
    # print("loss,similarity,binding_affinity,ba_target,bbb,bbb_target,seed_id,seed_ba,ref_id", file=fo1) 
    # write_file.write('loss,similarity,DAT_BA,NET_BA,SERT_BA,hERG_BA\n')
    for cy in range(args.cycle):
        #load reference    
        ref_id = randint(0,sref)   
        ref_tensor = torch.from_numpy(ref_ls[ref_id])
        ref_tensor = ref_tensor.float() 
        ref_tensor = ref_tensor.to(device)

        #load seed 
        # if args.init_vec == 'Provided':
        #     seed_id = randint(0,len(init_vec_ls))
        #     seed = init_vec_ls[seed_id]
        # else:
        #     seed = np.array([random()*2-1.0 for i in range(512)])   
        seed = np.array([random()*2-1.0 for i in range(512)])  
        seed_tensor = torch.from_numpy(seed)
        seed_tensor = seed_tensor.float() 
        seed_tensor = seed_tensor.to(device)    
        
        for epoch in range(1, args.epochs + 1):                
            alpha_list = [0.35,0.35,0.3]
            beta = 0.1
            vector,loss = generator_vector(args,ref_ls, seed_tensor, alpha_list, beta, target_sim)
            DAT_ba = DAT_predictor_model(vector)
            NET_ba = NET_predictor_model(vector)
            SERT_ba = SERT_predictor_model(vector)
            hERG_ba = hERG_predictor_model(vector)
            # # print(vector)
            # new_vector = vector.detach().cpu().numpy().reshape(1, -1)
            # new_vector.fillna(new_vector.mean())
            # hERG_ba = reg.predict(new_vector)

            if(DAT_ba <= args.output_lower_point and NET_ba <= args.output_lower_point and SERT_ba <= args.output_lower_point and hERG_ba >= args.output_upper_point):
                vector_list=torch.reshape(vector,(-1,)) 
                vector_list=vector_list.tolist()
                vector_lists.append(vector_list)                
                # print(vector_list, file=fo)                    
                sim = Tanimoto_coefficient(vector,ref_tensor)
                write_file.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(loss,sim,DAT_ba.float(),NET_ba.float(),SERT_ba.float(),hERG_ba.float()))

    df = pd.DataFrame(vector_lists)
    df.to_csv(str(args.save_optimized_vector_path), mode='a+', header=False)
        
if __name__ == '__main__':
    main()
