# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-05 00:52:42
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-05 01:59:29
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/solve_ode.py
Description: 
'''
from numpy.random import random
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 0.1
ref_coefficients = np.array([[0.35,0.35,0.3]]).reshape(-1,1)
refs = np.ones((3,512))
noise = np.array([random()*2-1.0 for i in range(512)])
init_vec = np.array([random()*2-1.5 for i in range(512)])
print(init_vec.shape)
print('=============================================')
# X = np.ones(512)
# print(noise.shape)
# a = alpha*X + np.sum(alpha*ref_coefficients*refs,axis=0) + noise
# print(a.shape)

# a= np.sum(alpha*ref_coefficients*refs,axis=0)
# print(a)

F = lambda t, X: alpha*X + np.sum(alpha*ref_coefficients*refs,axis=0) + noise

t_eval = np.arange(0, 2000.01, 0.01)
sol = solve_ivp(F, [0, 2000], init_vec, t_eval=t_eval)

print(sol.y)
print(sol.y[0])
print(sol.y[:,0].shape)
plt.figure(figsize = (12, 8))
plt.plot(sol.t, sol.y[0])
plt.xlabel('x')
# import pandas as pd
# # plt.savefig('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/images/ode.png',dpi=300)
# def read_latent_space(feature_file):
#     ''' Read data set in *.csv to data frame in Pandas'''
#     df_X = pd.read_csv(feature_file,header=None)
#     X = df_X.values # convert values in dataframe to numpy array (features)
#     return X
# init_vec_ls = read_latent_space('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/init_vec/init_vec.csv')
# print(init_vec_ls.shape)