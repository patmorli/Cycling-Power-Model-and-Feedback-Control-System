#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:52:38 2021

@author: patrickmayerhofer

PredictPower loads a dataset and runs an RNN on the 
input data to predict the power.
Looks at each trial individually
Specify filename and which variables you'd like to use as X and Y variable(s)'
"""

# libraries
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
import openloop_algorithms as oa
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import kerastuner
import pickle

"changeable variables"
#subject = range(11)
# try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
subject = [8]
trial = [1,2]
normalize_flag = 1
save_flag = 0
which_model = 'bayesiansearch' # 'hyperband', 'forloop', 'randomsearch', 'bayesiansearch'
# batch size: calculated based on lenght of trial

layers = [1] #[1,2,3,4]
nodes = [8, 16] #[8,16,32]

previous_flag = 0
X_variables = ['rpm_filtered','rpmdot_filtered'] #'rpmdot_filtered'
Y_variable = ['power_filtered']

tuner_epochs = 25
epochs = 10
max_trials = 10
patience = 1000
min_delta = 0.01

tuner_name = '20210505-195609' # if we want to use a specific tuner that has been developed earlier
previous_variable = Y_variable

#filename = 'Subject4_2'

"load summary file"
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/'
# excel summary file
dir_summary_file = dir_root + 'Summary.xlsx'
summary_file = pd.read_excel(dir_summary_file)
local_summary_file = summary_file # to change variables in the summary_file and save it later

"learning algorithm"
# callbacks
# Early Stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #min_delta=min_delta,
    patience=patience,
    #verbose=1, #will show progress bar
    #mode="auto",
    #baseline=None,
    restore_best_weights=True, #restores the weights of the best epoch not the last one (false)
)

cb_list  = [es] # es, tb

for subject_id in subject:
    for trial_id in trial:
        dir_save = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Individuals/'  \
            + which_model + '/Subject' + str(subject_id) + '/' + str(trial_id) + '/'
        # save for different nodes and layers and subjects
        save_r2_train = list()
        save_r2_test = list()
        save_rmse_train = list()
        save_rmse_test = list()
        save_data_train = list()
        save_data_test = list()
        save_trainPredict = list()
        save_testPredict = list()
        save_layers_nodes = list()
        
        """load and organize data. find increments, normalize,.."""
        increments, data = ppf.load_and_organize_data_individual(       \
            subject_id, trial_id, dir_root, normalize_flag)
        
            
      
        if which_model == 'forloop':
            # for the number of layers
            for layer in layers:
               
                # for the number of nodes 
                for node in nodes:
                    layers_nodes = [str(subject_id) + '_' + str(layer) + '_' + str(node)]
                    
                    print('layers: ' +  str(layer) + ', nodes: ' + str(node))
                    
                        
                    fivetrials_r2_train, fivetrials_r2_test,                \
                    fivetrials_rmse_train, fivetrials_rmse_test,            \
                    fivetrials_data_train, fivetrials_data_test,            \
                    fivetrials_trainPredict, fivetrials_testPredict =       \
                    oa.forloop_algorithm_individual(subject_id, trial_id,   \
                    layer, node, increments, data, X_variables, Y_variable, \
                    epochs, cb_list) 
                            
                    # save these results for respective layer and node number
                    save_layers_nodes.append(layers_nodes)
                    save_r2_train.append(fivetrials_r2_train)
                    save_r2_test.append(fivetrials_r2_test)
                    save_rmse_train.append(fivetrials_rmse_train)
                    save_rmse_test.append(fivetrials_rmse_test)
                    save_data_train.append(fivetrials_data_train)
                    save_data_test.append(fivetrials_data_test)
                    save_trainPredict.append(fivetrials_trainPredict)
                    save_testPredict.append(fivetrials_testPredict)
        elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
            """load and organize data"""   
            fivetrials_tuner_results = oa.kerastuneralgorithm_individual(   \
             data, X_variables, Y_variable, increments,                     \
             tuner_epochs, dir_save, cb_list, which_model  )
                    
                    # when trial of subject is done, save it                
        """
        if save_flag:
            save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Individuals/Subject' + str(subject_id) + '_' + str(trial_id) + '.pkl'
       
                
            with open(save_path, 'wb') as f:
                pickle.dump([save_r2_train, save_r2_test, 
                             save_rmse_train, save_rmse_test, 
                             save_data_train, save_data_test,
                             save_trainPredict, save_testPredict,
                             save_layers_nodes], f)    
        """    
                
        