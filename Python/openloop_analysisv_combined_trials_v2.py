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
import pandas as pd
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
import PredictPower_functions as ppf
import tensorflow as tf
import pickle
from kerastuner.engine.hyperparameters import HyperParameters
import openloop_algorithms as oa
from numpy import zeros, ones
import kerastuner
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import math


"changeable variables"
#subject = range(11)
# try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
subject = [8]
trial_id = [1,2]
normalize_flag = 1
save_flag = 1
which_model = 'bayesiansearch' # 'hyperband', 'forloop', 'randomsearch', 'bayesiansearch'
# batch size: calculated based on lenght of trial
"""
hp = HyperParameters()
hp.Choice('learning_rate', [0.001, 0.01])
hp.Int('hidden_layers', 1, 5, step = 1, default = 1)
hp.Int('units', 8, 64, step = 8, default = 8)
"""
layers = [1] #[1,2,3,4]
nodes = [8] #[8,16,32]

previous_flag = 0
X_variables = ['rpm_filtered', 'rpmdot_filtered', 'gear_ratio']
Y_variable = ['power_filtered']

tuner_epochs = 25
epochs = 10
max_trials = 10
patience = 5
min_delta = 0.001

fbracket = 39

#filename = 'Subject4_2'
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/'

"load summary file"
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
    dir_save = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Combined/'  \
    + which_model + '/Subject' + str(subject_id) + '/'
    """load and organize data"""
    increments, data = ppf.load_and_organize_data_combined(subject_id, trial_id, \
                  dir_root, normalize_flag, summary_file, fbracket)
    if which_model == 'forloop':
        """loading, preparing, training, and testing data for forloop model"""
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
        
        # for the number of layers
        for layer in layers:
          
           # for the number of nodes 
           for node in nodes:
               print('layers: ' +  str(layer) + ', nodes: ' + str(node))
               layers_nodes = [str(subject_id) + '_' + str(layer) + '_' + str(node)]
               """run full algorithm, not best function, but it gives better overview"""
               sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train,     \
               sixtrials_rmse_test, sixtrials_data_train, sixtrials_data_test,  \
               sixtrials_trainPredict, sixtrials_testPredict                    \
               = oa.forloop_algorithm_combined(subject_id, layer, node, increments,      \
                                      data, X_variables, Y_variable, epochs,    \
                                      cb_list)
                        
               # save these results for respective layer and node number
               save_layers_nodes.append(layers_nodes)
               save_r2_train.append(sixtrials_r2_train)
               save_r2_test.append(sixtrials_r2_test)
               save_rmse_train.append(sixtrials_rmse_train)
               save_rmse_test.append(sixtrials_rmse_test)
               save_data_train.append(sixtrials_data_train)
               save_data_test.append(sixtrials_data_test)
               save_trainPredict.append(sixtrials_trainPredict)
               save_testPredict.append(sixtrials_testPredict)
                
    
    
    
    elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
        """load and organize data"""
        increments, data = ppf.load_and_organize_data_combined(subject_id, trial_id, \
                 dir_root, normalize_flag, summary_file, fbracket)
        "run hyperband for all 6 increments"
        sixtrials_tuner_results = oa.kerastuneralgorithm_combined(       \
           data, X_variables, Y_variable, increments,           \
           tuner_epochs, dir_save, cb_list, which_model  )
      
        
        
                    
    if save_flag:
        if which_model == 'forloop':
            #save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/CombinedSubject' + str(subject_id) + '_' + '.pkl'
            save_path = dir_save + 'summary_results.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump([save_r2_train, save_r2_test, 
                                  save_rmse_train, save_rmse_test, 
                                  save_data_train, save_data_test,
                                  save_trainPredict, save_testPredict,
                                  save_layers_nodes, sixtrials_rmse_test,
                                  ], f)  
       
        elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
            #save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Hyperband/CombinedSubject' + str(subject_id) + '_' + '.pkl'
            save_path = dir_save + 'summary_results.pkl'
            
            with open(save_path, 'wb') as f:
                pickle.dump([sixtrials_tuner_results], f)  
    
    
   

                