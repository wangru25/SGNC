# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-12-16 14:30:22
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-18 22:55:34
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/generator.py
Description: 
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
from numpy.random import random
from numpy.random import randint
from torch.autograd import Variable  
from utils.predictor_network import *  

seed = int(sys.argv[1]) # 0
date = int(sys.argv[2]) # 20221209
optimized_vector_file = "results/generated_%d.csv"%date
ba_file = "results/evaluation_%d.csv"%date
os.system('python ./src/generator.py --save-optimized-vector-path %s --save-optimized-binding-affinity %s --seed %d'%(optimized_vector_file,ba_file,seed))

