# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2023-07-17 13:43:38
LastModifiedBy: Rui Wang
LastEditTime: 2023-07-17 15:06:25
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/test.py
Description: 
'''
import os
import sys
import torch
import numpy as np
import pandas as pd
from pysmiles import read_smiles
from molmass import Formula
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


def Tanimoto_coefficient(l1,l2):
    l1 = torch.from_numpy(l1).float()
    l2 = torch.from_numpy(l2).float()
    product=torch.mul(l1,l2)
    sum_product=torch.sum(product)
    l1_squared=torch.mul(l1,l1)
    sum_l1_squared=torch.sum(l1_squared)
    l2_squared=torch.mul(l2,l2)
    sum_l2_squared=torch.sum(l2_squared)
    tc=torch.div(sum_product,(sum_l1_squared+sum_l2_squared-sum_product)).numpy()
    return(tc)

a = np.array([1,2,4,5,6,7,8,9,10])
b = np.array([1,2,3,4,5,6,7,8,9])

a1 = Tanimoto_coefficient(a,b)
b1 = Similarities(a,b).tanimoto_similarity()
c1 = Similarities(a,b).cosine_similarity()
print(a1)
print(b1)
print(c1)