#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:06:46 2021

@author: patrickmayerhofer
"""

import numpy as np
import time
from numpy import zeros, ones
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
import PredictPower_functions as ppf
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import kerastuner
import pickle
import statistics

which_model = 'bayesiansearch'
subjects = [9]
trials = [1,2]


dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/ResultsPython/FindAlgorithm/Individuals/' + which_model + '/'


if which_model == 'bayesiansearch':
    # try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
    fivetrials_name = list()
    fivetrials_layers = list()
    fivetrials_units_0 = list()
    fivetrials_units_1 = list()
    fivetrials_units_2 = list()
    fivetrials_units_3 = list()
    fivetrials_units_4 = list()
    
    
    
    
    """load the data"""
    "changeable variables sub 1"
    
    for subject_id in subjects:
        for trial_id in trials:
            load_path = dir_root + '/Subject' + str(subject_id) + '/' + str(trial_id) + '/summary_results.pkl'
            
            with open(load_path, 'rb') as f:
                onetrials_name, onetrials_layers, onetrials_units_0, onetrials_units_1, onetrials_units_2, onetrials_units_3, onetrials_units_4 = pickle.load(f)  
            fivetrials_name = np.append(fivetrials_name, onetrials_name)  
            fivetrials_layers = np.append(fivetrials_layers, onetrials_layers) 
            fivetrials_units_0 = np.append(fivetrials_units_0, onetrials_units_0) 
            fivetrials_units_1 = np.append(fivetrials_units_1, onetrials_units_1) 
            fivetrials_units_2 = np.append(fivetrials_units_2, onetrials_units_2) 
            fivetrials_units_3 = np.append(fivetrials_units_3, onetrials_units_3) 
            fivetrials_units_4 = np.append(fivetrials_units_4, onetrials_units_4) 