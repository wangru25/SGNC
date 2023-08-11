# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2023-01-09 10:22:44
LastModifiedBy: Rui Wang
LastEditTime: 2023-01-09 10:40:02
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/statistic_drugbank.py
Description: 
'''
import numpy as np
import pandas as pd
from collections import Counter
from pysmiles import read_smiles


def mol_mass(smi):
    mol = read_smiles(smi, explicit_hydrogen=True)
    graph_mol = mol.nodes(data='element')
    num_atoms = len(graph_mol)
    elements = []
    for i in range(num_atoms):
        elements.append(graph_mol[i])
    return elements
    
smi_df = pd.read_csv('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/drugbank/full_drugbank.smi',header=None)
all_elements = []
for i in range(smi_df.shape[0]):
    smi = smi_df[0][i]
    elements = mol_mass(smi)
    all_elements += elements

results= Counter(all_elements)
print(results)

# elements = mol_mass('CC1=CC=CC=C1C1=NNC(=S)N1N')
# print(elements)