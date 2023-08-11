# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-27 16:15:12
LastModifiedBy: Rui Wang
LastEditTime: 2023-08-09 22:52:44
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/select_reference.py
Description: select smi with 1)high binding 2)low molecular weight 3) high similarity with others in the same dataset.
'''
import os
import sys
import torch
import numpy as np
import pandas as pd
from pysmiles import read_smiles
from molmass import Formula

class Similarities:
    '''
    Ref: https://docs.eyesopen.com/toolkits/python/graphsimtk/measure.html#figure-measure-fptanimoto
    '''
    def __init__(self, A, B):
        # A and B are both numpy array
        self.A = torch.from_numpy(A)
        self.B = torch.from_numpy(B)

    def dot_AB(self):
        # number of bits set “on” in both fingerprints
        return torch.mul(self.A, self.B)
    
    def L2_norm_A(self):
        return torch.norm(self.A)

    def L2_norm_B(self):
        return torch.norm(self.B)

    def cosine_similarity(self):
        return torch.div(self.dot_AB, self.L2_norm_A * self.L2_norm_B).numpy()

    def tanimoto_similarity(self):
        return torch.div(self.dot_AB, self.L2_norm_A + self.L2_norm_B - self.dot_AB).numpy()

class Similarities:
    def __init__(self, A, B):
        # A and B are both numpy arrays
        self.A = torch.from_numpy(A).float()
        self.B = torch.from_numpy(B).float()

    def dot_AB(self):
        # dot product of A and B
        return torch.sum(torch.mul(self.A, self.B))
    
    def L2_norm_A(self):
        # squared L2 norm of A
        return torch.sum(torch.mul(self.A, self.A))

    def L2_norm_B(self):
        # squared L2 norm of B
        return torch.sum(torch.mul(self.B, self.B))

    def cosine_similarity(self):
        return (self.dot_AB() / torch.sqrt(self.L2_norm_A() * self.L2_norm_B())).numpy()

    def tanimoto_similarity(self):
        return (self.dot_AB() / (self.L2_norm_A() + self.L2_norm_B() - self.dot_AB())).numpy()

class MolMass:
    __MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                    'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                    'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                    'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                    'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                    'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                    'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                    'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                    'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                    'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                    'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                    'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                    'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                    'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                    'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                    'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                    'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                    'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                    '': 0}
                    
    def __init__(self):
        pass

    def mol_mass(self, smi):
        mol = read_smiles(smi, explicit_hydrogen=True)
        graph_mol = mol.nodes(data='element')
        num_atoms = len(graph_mol)
        total_mass = 0.0
        for i in range(num_atoms):
            total_mass += self.__MM_of_Elements[graph_mol[i]]
        return total_mass

def get_properties():
    datasets = ['DAT','SERT','NET','extended_hERG']
    for dataset in datasets:
        data_dir = f'../data/{dataset}/'
        save_dir = f'../results/'
        smi_df = pd.read_table(data_dir + f'{dataset}.smi', header=None)
        ls_npy = pd.read_csv(data_dir + f'ls-{dataset}.csv', header=None).to_numpy()
        label_df = pd.read_csv(data_dir + 'y_train.csv', header=None)
        write_file = open(save_dir + f'{dataset}_properties.csv','w')
        write_file.write('smiles,binding,mol_mass\n')

        for i in range(len(smi_df)):
            smi = smi_df[0][i]
            ba = label_df[0][i]
            mol_mass = MolMass().mol_mass(smi)
            write_file.write(f'{smi},{ba},{mol_mass}\n')
            # for j in range(len(smi_df)):
            #     vec_i = ls_npy[i]
            #     vec_j = ls_npy[j]
            #     cosine_similarity = Similarities(vec_i,vec_j).cosine_similarity()
        write_file.close()

def get_range():
    datasets = ['DAT','SERT','NET','extended_hERG']
    for dataset in datasets:
        result_df = pd.read_csv(f'../results/{dataset}_properties.csv')
        ba_list = result_df['binding'].to_list()
        mol_mass_list = result_df['mol_mass'].to_list()
        print(f'{dataset}============ \n The range of binding is [{min(ba_list)},{max(ba_list)}]\n The range of mol mass is [{min(mol_mass_list)},{max(mol_mass_list)}]')

def get_ref_similarity():
    datasets = ['DAT','SERT','NET']
    ref_npy = pd.read_csv(f'../data/reference/ls-reference.csv',header=None).to_numpy()
    for idx, dataset in enumerate(datasets):
        data_dir = f'../data/{dataset}/'
        ls_npy = pd.read_csv(data_dir + f'ls-{dataset}.csv', header=None).to_numpy()
        ref_vec = ref_npy[idx]
        total_sim = 0
        big_sim = 0
        for i in range(ls_npy.shape[0]):
            vec_i = ls_npy[i]
            cosine_similarity = Similarities(vec_i,ref_vec).cosine_similarity()
            total_sim += cosine_similarity
            if cosine_similarity >= 0.7:
                big_sim += 1
        print(dataset, (total_sim-1)/ls_npy.shape[0], big_sim)


    


if __name__ == '__main__':
    # get_properties()
    # get_range()
    get_ref_similarity()