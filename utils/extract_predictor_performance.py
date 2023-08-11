# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2023-07-07 13:17:36
LastModifiedBy: Rui Wang
LastEditTime: 2023-07-07 13:50:20
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/extract_predictor_performance.py
Description: 
'''
import os
import sys
import numpy as np

def extract_predictor_performance_ANN(file_dir):
    txt_file = open(file_dir, 'r')
    PCC = 0
    RMSE = 0
    for line in txt_file:
        if '[Epoch: 1000]' in line:
            pcc = float(line.split('PCC: ')[1].split(' ')[0][:6])
            rmse = float(line.split('RMSE: ')[1].split(' ')[0][:6])
            PCC += pcc
            RMSE += rmse
    return PCC/10, RMSE/10

def extract_predictor_performance_GBDT(file_dir):
    txt_file = open(file_dir, 'r')
    PCC = 0
    RMSE = 0
    for line_num, line in enumerate(txt_file):
        if str(line[0].split(' ')[0]) in ['0','1','2','3','4','5','6','7','8','9'] and line_num > 1:
            pcc = float(line.split(' ')[1])
            rmse = float(line.split(' ')[1])
            PCC += pcc
            RMSE += rmse
    return PCC/10, RMSE/10

if __name__ == "__main__":
    for dataset in sys.argv[1:]:
        file_dir = '../sbatch/%s_predictor_ANN_kfold.out'%dataset
        file_dir_2 = '../sbatch/%s_predictor_GBDT_kfold.out'%dataset
        PCC_ANN, RMSE_ANN = extract_predictor_performance_ANN(file_dir)
        PCC_GBDT, RMSE_GBDT = extract_predictor_performance_GBDT(file_dir_2)
        print("%s ANN PCC = %.4f, RMSE = %.4f"%(dataset,PCC_ANN,RMSE_ANN))
        print("%s GBDT PCC = %.4f, RMSE = %.4f"%(dataset,PCC_GBDT,RMSE_GBDT))
