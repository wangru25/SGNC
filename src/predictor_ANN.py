# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-20 22:59:50
LastModifiedBy: Rui Wang
LastEditTime: 2023-07-21 02:14:27
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/predictor_ANN.py
Description: 
'''
import sys
import argparse
import numpy as np
import pandas as pd
from numpy.random import random
from numpy.random import randint
from torch.autograd import Variable  
from sklearn.model_selection import KFold
sys.path.append('../')
from utils import * 
#os.environ['CUDA_VISIBLE_DEVICES'] = args.device

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ===============================Data Preprocessing==============================
def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file, header=None)
    df_y = pd.read_csv(label_file, header=None)
    X = df_X.values
    y = df_y.values
    return X, y
    
# ====================================Functions==================================
def RMSE(ypred, yexact):
    return torch.sqrt(torch.sum((ypred-yexact)**2)/ypred.shape[0])

def PCC(ypred, yexact):
    from scipy import stats
    a = (Variable(yexact).data).cpu().numpy().ravel()
    b = (Variable(ypred).data).cpu().numpy().ravel()
    pcc = stats.pearsonr(a,b)
    return pcc

# =================================Training & Testing============================
def predictor_train(args, model, device, train_loader, optimizer, epoch): #low energy generating
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output.float(), target.float()) 
        loss.backward()
        optimizer.step()
        # return(loss.item())

def predictor_test(args, model, device, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output.float(), target.float(), reduction='sum').item() # sum up batch loss. For regression
            pcc = PCC(output.float(), target.float())[0]  # For regression
            rmse = RMSE(output.float(), target.float())   # For regression
    test_loss /= len(test_loader.dataset)
    if epoch % args.log_interval == 0:
        print("[test_loss: {:.4f}] [PCC: {:.4f}] [RMSE: {:.4f}] [Epoch: {:d}] [ST] ".format(test_loss, pcc, rmse, epoch)) # For regression


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Predictor Model')
    parser.add_argument('--dataset',default="GRK5", type=str,
                            help="dataset id")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                            help='input batch size for testing (default: 50)')  
    parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                            help='latent_size')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                            help='number of epochs to generate (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                            help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.3, metavar='M',
                            help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    parser.add_argument('--device',default="1", type=str,
                            help="number of cuda visible devise")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                            help='For Saving the current Model')
    parser.add_argument('--is-kfold',default=0, type=int,
                            help="is k-fold cross validation?")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # =================================Load Data=================================
    latent_space_file = "../data/%s/ls-%s.csv"%(args.dataset, args.dataset)
    latent_space_file = "/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/init_vec/ls-init_vec.csv"
    label_file = "../data/%s/y_train.csv"%(args.dataset)
    x_train, y_train = read_dataset(latent_space_file, label_file)
    x_test, y_test = read_dataset(latent_space_file, label_file)
    
    # predictor_network
    dropout=0.0 # default=0.3   
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
    DAT_predictor_model.load_state_dict(torch.load('../model/DAT_predictor_model.pt'))
    DAT_predictor_model.eval()

    NET_predictor_model.load_state_dict(torch.load('../model/NET_predictor_model.pt'))
    NET_predictor_model.eval()

    SERT_predictor_model.load_state_dict(torch.load('../model/SERT_predictor_model.pt'))
    SERT_predictor_model.eval()
    
    hERG_predictor_model.load_state_dict(torch.load('../model/extended_hERG_predictor_model.pt'))
    hERG_predictor_model.eval()

    test_data = torch.from_numpy(x_test).float().to(device)
    output_DAT = DAT_predictor_model(test_data).detach().cpu().numpy()
    output_NET = NET_predictor_model(test_data).detach().cpu().numpy()
    output_SERT = SERT_predictor_model(test_data).detach().cpu().numpy()
    output_hERG = hERG_predictor_model(test_data).detach().cpu().numpy()
    print(output_DAT, output_NET, output_SERT, output_hERG)

    # x = []
    # kf = KFold(n_splits=10, shuffle=True)
    # for id, (train, test) in enumerate(kf.split(x_train)):
    #     x_train_fold, y_train_fold = x_train[train], y_train[train]
    #     x_test_fold, y_test_fold = x_train[test], y_train[test]

    #     test_data = torch.from_numpy(x_test_fold).float().to(device)
    #     output = hERG_predictor_model(test_data).detach().cpu().numpy()
    #     x.append(output)

    # combined_array = np.concatenate([t for t in x], axis=0)
    # print(combined_array.shape)
    # df = pd.DataFrame(combined_array)
    # df.to_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/extended_hERG/y_predicted.csv', index=False,header=None)


    # if args.is_kfold == 1: #is_kfold is True
    #     kf = KFold(n_splits=10, shuffle=True)
    #     for id, (train, test) in enumerate(kf.split(x_train)):
    #         x_train_fold, y_train_fold = x_train[train], y_train[train]
    #         x_test_fold, y_test_fold = x_train[test], y_train[test]
    #         train_data = torch.from_numpy(x_train_fold).float()
    #         test_data = torch.from_numpy(x_test_fold).float()
    #         trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train_fold.reshape(-1,1)))
    #         testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test_fold.reshape(-1,1)))
    #         train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #         test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=3000, shuffle=False)
    #         # =================================Design Net================================
    #         dropout = 0.0
    #         # network_layers = [1024,1536,1536,1024]
    #         network_layers = [512,1024,512]
    #         in_dim = args.latent_size
            
    #         if(len(network_layers)==1):
    #             predictor_model = Net1(in_dim,network_layers[0],dropout).to(device)
    #         elif(len(network_layers)==2):
    #             predictor_model = Net2(in_dim,network_layers[0],network_layers[1],dropout).to(device)
    #         elif(len(network_layers)==3):
    #             predictor_model = Net3(in_dim,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
    #         elif(len(network_layers)==4):
    #             predictor_model = Net4(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],dropout).to(device)
    #         elif(len(network_layers)==5):
    #             predictor_model = Net5(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4],dropout).to(device)
    #         elif(len(network_layers)==6):
    #             predictor_model = Net6(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4], network_layers[5], dropout).to(device)

    #         optimizer = optim.Adam(predictor_model.parameters(), lr=args.lr, eps=1e-08, amsgrad=False)
    #         lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.1, last_epoch = -1)
    #         for epoch in range(1, args.epochs + 1):
    #             predictor_train(args, predictor_model, device, train_loader, optimizer, epoch)
    #             predictor_test(args, predictor_model, device, epoch, test_loader)
    #             # lr_adjust.step()
    # else:
    #     train_data = torch.from_numpy(x_train).float()
    #     test_data = torch.from_numpy(x_test).float()
    #     trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(y_train.reshape(-1,1)))
    #     testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(y_test.reshape(-1,1)))
    #     train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    #     # =================================Design Net================================
    #     dropout = 0.0
    #     # network_layers = [1024,1536,1536,1024]
    #     network_layers = [512,1024,512]
    #     in_dim = args.latent_size
        
    #     if(len(network_layers)==1):
    #         predictor_model = Net1(in_dim,network_layers[0],dropout).to(device)
    #     elif(len(network_layers)==2):
    #         predictor_model = Net2(in_dim,network_layers[0],network_layers[1],dropout).to(device)
    #     elif(len(network_layers)==3):
    #         predictor_model = Net3(in_dim,network_layers[0],network_layers[1],network_layers[2],dropout).to(device)
    #     elif(len(network_layers)==4):
    #         predictor_model = Net4(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],dropout).to(device)
    #     elif(len(network_layers)==5):
    #         predictor_model = Net5(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4],dropout).to(device)
    #     elif(len(network_layers)==6):
    #         predictor_model = Net6(in_dim,network_layers[0],network_layers[1],network_layers[2],network_layers[3],network_layers[4], network_layers[5], dropout).to(device)

    #     optimizer = optim.Adam(predictor_model.parameters(), lr=args.lr, eps=1e-08, amsgrad=False)
    #     lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.1, last_epoch = -1)
    #     for epoch in range(1, args.epochs + 1):
    #         predictor_train(args, predictor_model, device, train_loader, optimizer, epoch)
    #         output = predictor_test(args, predictor_model, device, epoch, test_loader)
    #         # lr_adjust.step()
    #     if (args.save_model):
    #         torch.save(predictor_model.state_dict(),"../model/%s_predictor_model.pt"%args.dataset)

if __name__ == '__main__':
    main()