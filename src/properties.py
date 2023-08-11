# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-22 16:41:36
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-22 17:31:18
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/properties.py
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

date = int(sys.argv[1])
ADMET_df = pd.read_csv('./results/generator_%d/ADMET.csv'%date)
write_selected = open('./results/generator_%d/0_final_selected.csv'%date,'w')
write_selected.write('smiles,FDAMDD,F(20%),LogP,LogS,T12,Caco-2,SAS\n')
for i in range(len(ADMET_df)):
    sas = sascorer.calculateScore(Chem.MolFromSmiles(ADMET_df['smiles'][i]))
    print(ADMET_df['smiles'][i],ADMET_df['FDAMDD'][i],ADMET_df['F(20%)'][i],ADMET_df['LogP'][i],ADMET_df['LogS'][i],ADMET_df['T12'][i],ADMET_df['Caco-2'][i],sas)
    print('=========================================')
    if ADMET_df['FDAMDD'][i] <= 0.7 and ADMET_df['F(20%)'][i]  <= 0.7 and ADMET_df['LogP'][i]  <= 3 and ADMET_df['LogS'][i]  >= -4 and ADMET_df['LogS'][i]  <= 0.5 and ADMET_df['T12'][i] <= 0.7 and ADMET_df['Caco-2'][i] > -5.15 and sas < 6:
        # print(ADMET_df['smiles'][i],ADMET_df['F(20%)'][i],ADMET_df['LogP'][i],ADMET_df['LogS'][i],ADMET_df['T12'][i],ADMET_df['Caco-2'][i],sas)
        write_selected.write('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(ADMET_df['smiles'][i],ADMET_df['FDAMDD'][i],ADMET_df['F(20%)'][i],ADMET_df['LogP'][i],ADMET_df['LogS'][i],ADMET_df['T12'][i],ADMET_df['Caco-2'][i],sas))
