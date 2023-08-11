# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-04 11:54:52
LastModifiedBy: Rui Wang
LastEditTime: 2022-11-04 11:54:53
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/predictor_network_multitask.py
Description: 
'''
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms


class MT_Net1(nn.Module):
    def __init__(self, input_dim, neurons1,dropout):
        super(MT_Net1, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MT_Net2(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2,dropout):
        super(MT_Net2, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.neurons2=neurons2
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

class MT_Net3(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, neurons3, dropout):
        super(MT_Net3, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.neurons2=neurons2
        self.neurons3=neurons3
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, self.neurons3)
        self.fc4 = nn.Linear(self.neurons3, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)
        return x

class MT_Net4(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, neurons3, neurons4, dropout):
        super(MT_Net4, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.neurons2=neurons2
        self.neurons3=neurons3
        self.neurons4=neurons4
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, self.neurons3)
        self.fc4 = nn.Linear(self.neurons3, self.neurons4)
        self.fc5 = nn.Linear(self.neurons4, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        x = self.fc5(x)
        return x
    
class MT_Net5(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, neurons3, neurons4, neurons5, dropout):
        super(MT_Net5, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.neurons2=neurons2
        self.neurons3=neurons3
        self.neurons4=neurons4
        self.neurons5=neurons5
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, self.neurons3)
        self.fc4 = nn.Linear(self.neurons3, self.neurons4)
        self.fc5 = nn.Linear(self.neurons4, self.neurons5)
        self.fc6 = nn.Linear(self.neurons5, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        x = F.relu(self.fc5(x))
        x = self.drop(x)
        x = self.fc6(x)
        return x
    
class MT_Net6(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, neurons3, neurons4, neurons5, neurons6, dropout):
        super(MT_Net6, self).__init__()
        self.input_dim=input_dim
        self.neurons1=neurons1
        self.neurons2=neurons2
        self.neurons3=neurons3
        self.neurons4=neurons4
        self.neurons5=neurons5
        self.neurons6=neurons6
        self.dropout=dropout
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, self.neurons3)
        self.fc4 = nn.Linear(self.neurons3, self.neurons4)
        self.fc5 = nn.Linear(self.neurons4, self.neurons5)
        self.fc6 = nn.Linear(self.neurons5, self.neurons6)
        self.fc7 = nn.Linear(self.neurons6, 4)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        x = F.relu(self.fc5(x))
        x = self.drop(x)
        x = F.relu(self.fc6(x))
        x = self.drop(x)
        x = self.fc7(x)
        return x