#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:34:20 2021

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

"changeable variables"
#subject = range(11)
subject_id = 11
trial_id = 1
fold = 1
nn_architecture = 0 # this is the index. look up which one is which architecture

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/'
load_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Subject' + str(subject_id) + '_' + str(trial_id) + '.pkl'


with open(load_path, 'rb') as f:
           save_r2_train, save_r2_test, save_rmse_train, save_rmse_test,save_data_train, save_data_test, save_trainPredict, save_testPredict, save_layers_nodes  =  pickle.load(f)


power_measured = save_data_test[nn_architecture][fold].power_filtered
power_measured = power_measured.reset_index(drop = True)
power_predicted = save_testPredict[nn_architecture][fold][:,0,0]

plt.figure()
plt.title('Predicted Power vs Actual Power normalized-test')
predicted_power = plt.plot(power_predicted , label = 'predicted_power', color = 'r')
actual_power = plt.plot(power_measured, label = 'measured_power', color = 'b')
plt.legend()
