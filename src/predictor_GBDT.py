# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-20 22:53:49
LastModifiedBy: Rui Wang
LastEditTime: 2022-11-20 23:37:25
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/src/predictor_GBDT.py
Description: 
'''
import sys
import pandas as pd
import numpy as np

seed = int(sys.argv[1])
dataset = sys.argv[2]
is_kfold = int(sys.argv[3]) # 0 is False, 1 is True

from sklearn.model_selection import KFold
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
    return np.sqrt(np.sum((ypred.ravel()-yexact.ravel())**2)/ypred.shape[0])

def PCC(ypred, yexact):
    from scipy import stats
    a = yexact.ravel()
    b = ypred.ravel()
    pcc = stats.pearsonr(a,b)
    return pcc

# ===================================GBDT=======================================
def random_forest_model(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    regr = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=10000, max_depth=8, min_samples_split=3, subsample=0.3, max_features='sqrt', random_state = seed)
    regr.fit(X_train, y_train.ravel())
    y_pred = regr.predict(X_test)
    return y_pred,regr

x_train, y_train = read_dataset('../data/%s/ls-%s.csv'%(dataset,dataset), '../data/%s/y_train.csv'%dataset)

if is_kfold == 1:
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for id, (train, test) in enumerate(kf.split(x_train)):
        x_train_fold, y_train_fold = x_train[train], y_train[train]
        x_test_fold, y_test_fold = x_train[test], y_train[test]
        y_pred, regr = random_forest_model(x_train_fold, y_train_fold, x_test_fold, y_test_fold)
        pearson = PCC(y_pred,y_test_fold)[0]
        error = RMSE(y_pred,y_test_fold)
        print(id,pearson, error)
else:
    y_pred, regr = random_forest_model(x_train, y_train, x_train, y_train)
    pearson = PCC(y_pred,y_train)[0]
    error = RMSE(y_pred,y_train)
    print(pearson, error)
    import pickle
    with open('../model/%s_GBDT.pkl'%dataset,'wb') as f:
        pickle.dump(regr,f)
