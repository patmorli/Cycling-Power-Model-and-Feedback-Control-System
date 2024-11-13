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

subject = [9, 11, 12]
which_model = 'bayesiansearch'


# try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
r2_train_mean = list()
r2_train_stdev = list()
r2_test_mean = list()
r2_test_stdev = list()
rmse_train_mean = list()
rmse_train_stdev = list()
rmse_test_mean = list()
rmse_test_stdev = list()
sub_unit0_unitn_learningrate = list()


dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/' +   \
'/OpenLoop/ResultsPython/FindAlgorithm/Combined/'


"""load the data"""
"changeable variables sub 1"

for subject_id in subject:
    load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/summary_results.pkl'
    with open(load_path, 'rb') as f:
        sixtrials_hyperband_results = pickle.load(f)   
        
        # look into which results won most often
        for i in range(len(sixtrials_r2_train)):
            best_hps = sixtrials_hyperband_results[i].get_best_hyperparameters(1)[0]
            units_0 = best_hps.get('units_0')
            learning_rate = best_hps.get('learning_rate')
            num_layers = best_hps.get('num_layers') - 1 
            units_1 = 0
            units_2 = 0
            units_3 = 0
            units_4 = 0
            units_5 = 0
                
            if num_layers >= 1:
                units_1 = best_hps.get('units_1')
            if num_layers >= 2:
                units_2 = best_hps.get('units_2')
            if num_layers >= 3:
                units_3 = best_hps.get('units_3')  
            if num_layers >= 4: 
                units_4 = best_hps.get('units_4')
            if num_layers >= 5:
                units_5 = best_hps.get('units_5')
                
            name = str(subject_id) + '_' + str(i) + '_' + str(units_0) +    \
                '_' + str(units_1) + '_' + str(units_2) +                   \
                '_' + str(units_3) + '_' + str(units_4) +                   \
                '_' + str(units_5) + '_' + str(learning_rate)
                
            sub_unit0_unitn_learningrate.append(name)    
                
                
            
                    
            
            
        
        
        
        
        for i in range(len(sixtrials_r2_train)):
              r2_train_mean.append(statistics.mean(sixtrials_r2_train[i]))
              r2_train_stdev.append(statistics.stdev(sixtrials_r2_train[i]))
              r2_test_mean.append(statistics.mean(sixtrials_r2_test[i]))
              r2_test_stdev.append(statistics.stdev(sixtrials_r2_test[i]))
              rmse_train_mean.append(statistics.mean(sixtrials_rmse_train[i]))
              rmse_train_stdev.append(statistics.stdev(sixtrials_rmse_train[i]))
              rmse_test_mean.append(statistics.mean(sixtrials_rmse_test[i]))
              rmse_test_stdev.append(statistics.stdev(sixtrials_rmse_test[i]))
              
              sub_trail_layer_node.append(save_layers_nodes[i])
      
"changeable variables sub 8"
#subject = range(11)
subject = [8]
trial = [1,2]
for subject_id in subject:
    for trial_id in trial:
        load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/summary_results.pkl'       
        
        with open(load_path, 'rb') as f:
                   #save_r2_train, save_r2_test, save_rmse_train, save_rmse_test,save_data_train, save_data_test, save_trainPredict, save_testPredict, save_layers_nodes  =  pickle.load(f)   
                   sixtrials_name  =  pickle.load(f)   

        
        
        
        for i in range(len(save_r2_train)):
              r2_train_mean.append(statistics.mean(save_r2_train[i]))
              r2_train_stdev.append(statistics.stdev(save_r2_train[i]))
              r2_test_mean.append(statistics.mean(save_r2_test[i]))
              r2_test_stdev.append(statistics.stdev(save_r2_test[i]))
              rmse_train_mean.append(statistics.mean(save_rmse_train[i]))
              rmse_train_stdev.append(statistics.stdev(save_rmse_train[i]))
              rmse_test_mean.append(statistics.mean(save_rmse_test[i]))
              rmse_test_stdev.append(statistics.stdev(save_rmse_test[i]))
              sub_trail_layer_node.append(save_layers_nodes[i])  
              
"changeable variables sub 11"
#subject = range(11)
subject = [11]
trial = [1,2,3]
for subject_id in subject:
    for trial_id in trial:
        load_path = dir_root + 'Subject' + str(subject_id) + '_' + str(trial_id) + '.pkl'
        
        #load 
        with open(load_path, 'rb') as f:
                   save_r2_train, save_r2_test, save_rmse_train, save_rmse_test,save_data_train, save_data_test, save_trainPredict, save_testPredict, save_layers_nodes  =  pickle.load(f)   
        
        
        
        # calculate mean 
        for i in range(len(save_r2_train)):
              r2_train_mean.append(statistics.mean(save_r2_train[i]))
              r2_train_stdev.append(statistics.stdev(save_r2_train[i]))
              r2_test_mean.append(statistics.mean(save_r2_test[i]))
              r2_test_stdev.append(statistics.stdev(save_r2_test[i]))
              rmse_train_mean.append(statistics.mean(save_rmse_train[i]))
              rmse_train_stdev.append(statistics.stdev(save_rmse_train[i]))
              rmse_test_mean.append(statistics.mean(save_rmse_test[i]))
              rmse_test_stdev.append(statistics.stdev(save_rmse_test[i]))
              sub_trail_layer_node.append(save_layers_nodes[i])   
              
"changeable variables sub 12"
#subject = range(11)
subject = [12]
trial = [1,2]
for subject_id in subject:
    for trial_id in trial:
        load_path = dir_root + 'Subject' + str(subject_id) + '_' + str(trial_id) + '.pkl'
        
        
        with open(load_path, 'rb') as f:
                   save_r2_train, save_r2_test, save_rmse_train, save_rmse_test,save_data_train, save_data_test, save_trainPredict, save_testPredict, save_layers_nodes  =  pickle.load(f)   
        
        
        
        
        for i in range(len(save_r2_train)):
              r2_train_mean.append(statistics.mean(save_r2_train[i]))
              r2_train_stdev.append(statistics.stdev(save_r2_train[i]))
              r2_test_mean.append(statistics.mean(save_r2_test[i]))
              r2_test_stdev.append(statistics.stdev(save_r2_test[i]))
              rmse_train_mean.append(statistics.mean(save_rmse_train[i]))
              rmse_train_stdev.append(statistics.stdev(save_rmse_train[i]))
              rmse_test_mean.append(statistics.mean(save_rmse_test[i]))
              rmse_test_stdev.append(statistics.stdev(save_rmse_test[i]))
              sub_trail_layer_node.append(save_layers_nodes[i])                
              
data = pd.DataFrame(list(zip(sub_trail_layer_node, 
                             r2_test_mean, r2_test_stdev, 
                               r2_train_mean, r2_train_stdev,
                             rmse_test_mean, rmse_test_stdev,
                             rmse_train_mean, rmse_train_stdev)),
                    columns = ['file', 
                               'r2_test_mean', 'r2_test_stdev', 
                               'r2_train_mean', 'r2_train_stdev',
                               'rmse_test_mean', 'rmse_test_stdev',
                               'rmse_train_mean', r'mse_train_stdev']) 
dir_save = dir_root + 'FindAlgorithm_Summary.csv'
data.to_csv(dir_save)        
  