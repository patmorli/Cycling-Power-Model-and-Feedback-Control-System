#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:54:00 2022

@author: patrickmayerhofer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
import PredictPower_functions as ppf
import pickle
import os

subject=[2,3,4,5,6,7,8,9,10,11,12,13]
trials = [1,2]
#subject = [9,11,12]
which_model = 'custom' # beayesiansearch, custom
bayesiansearch_model_name = 'novdot_1_16_1024_16'
plot_flag = 0 # creates hundreds of plots when running through (not sure why yet) make breakpoint for now
save_flag = 1 
single_model_save_flag = 1 # if only checking one single model and 

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData' +   \
    '/OpenLoop/ResultsPython/FindAlgorithm/Individuals/'

if which_model == 'custom':
    
    tested_models_r2_train_mean = list()
    tested_models_r2_test_mean = list()
    tested_models_r2_alternative_train_mean = list()
    tested_models_r2_alternative_test_mean = list()
    tested_models_rmse_train_mean = list()
    tested_models_rmse_test_mean = list()
    tested_models_nrmse_train_mean = list()
    tested_models_nrmse_test_mean = list()
    tested_models_norm_mean_err_train_mean = list()
    tested_models_norm_mean_err_test_mean = list()
    
    tested_models_r2_train_std = list()
    tested_models_r2_test_std = list()
    tested_models_r2_alternative_train_std = list()
    tested_models_r2_alternative_test_std = list()
    tested_models_rmse_train_std = list()
    tested_models_rmse_test_std = list()
    tested_models_nrmse_train_std = list()
    tested_models_nrmse_test_std = list()
    tested_models_norm_mean_err_train_std = list()
    tested_models_norm_mean_err_test_std = list()
    
    load_path = dir_root + which_model + '/Subject9/1/'
    tested_models = [name for name in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, name))]
    tested_models = ['[8]0.01']
    
    all_r2_train_mean = list()
    all_r2_test_mean = list()
    all_r2_alternative_train_mean = list()
    all_r2_alternative_test_mean = list()
    all_rmse_train_mean = list()
    all_rmse_test_mean = list()
    all_nrmse_train_mean = list()
    all_nrmse_test_mean = list()
    all_norm_mean_err_train_mean = list()
    all_norm_mean_err_test_mean = list()
    
    all_r2_train_std = list()
    all_r2_test_std = list()
    all_r2_alternative_train_std = list()
    all_r2_alternative_test_std = list()
    all_rmse_train_std = list()
    all_rmse_test_std = list()
    all_nrmse_train_std = list()
    all_nrmse_test_std = list()
    all_norm_mean_err_train_std = list()
    all_norm_mean_err_test_std = list()
    
    all_loss = list()
    all_val_loss = list() 
    all_belonging_name = list()
    
    
    
    for model in tested_models:
        subject_r2_train_mean = list()
        subject_r2_test_mean = list()
        subject_r2_alternative_train_mean = list()
        subject_r2_alternative_test_mean = list()
        subject_rmse_train_mean = list()
        subject_rmse_test_mean = list()
        subject_nrmse_train_mean = list()
        subject_nrmse_test_mean = list()
        subject_norm_mean_err_train_mean = list()
        subject_norm_mean_err_test_mean = list()
        
        subject_r2_train_std = list()
        subject_r2_test_std = list()
        subject_r2_alternative_train_std = list()
        subject_r2_alternative_test_std = list()
        subject_rmse_train_std = list()
        subject_rmse_test_std = list()
        subject_nrmse_train_std = list()
        subject_nrmse_test_std = list()
        subject_norm_mean_err_train_std = list()
        subject_norm_mean_err_test_std = list()
        
        for subject_id in subject:
            trial_r2_train_mean = list()
            trial_r2_test_mean = list()
            trial_r2_alternative_train_mean = list()
            trial_r2_alternative_test_mean = list()
            trial_rmse_train_mean = list()
            trial_rmse_test_mean = list()
            trial_nrmse_train_mean = list()
            trial_nrmse_test_mean = list()
            trial_norm_mean_err_train_mean = list()
            trial_norm_mean_err_test_mean = list()
            
            trial_r2_train_std = list()
            trial_r2_test_std = list()
            trial_r2_alternative_train_std = list()
            trial_r2_alternative_test_std = list()
            trial_rmse_train_std = list()
            trial_rmse_test_std = list()
            trial_nrmse_train_std = list()
            trial_nrmse_test_std = list()
            trial_norm_mean_err_train_std = list()
            trial_norm_mean_err_test_std = list()
            
            for trial_id in trials:
                load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/' + str(trial_id) + '/' + model + '/'
              
                with open(load_path + 'summary_results.pkl', 'rb') as f:
                     sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train, sixtrials_rmse_test, sixtrials_nrmse_train, sixtrials_nrmse_test, sixtrials_data_train, sixtrials_data_test, sixtrials_trainPredict, sixtrials_testPredict, sixtrials_loss = pickle.load(f)  
                     #r2, rmse, nrmse = ppf.cost(sixtrials_data_test[0].power_filtered[7:], sixtrials_testPredict[0])
                     if 1:
                         r2_alternative_train = list()
                         r2_alternative_test = list()
                         norm_mean_err_train = list()
                         norm_mean_err_test = list()
                         for looper in range(len(sixtrials_data_test)):
                             r2_alternative_train.append(ppf.r2_alternative(sixtrials_data_train[looper].power_filtered[7:], sixtrials_trainPredict[looper]))
                             r2_alternative_test.append(ppf.r2_alternative(sixtrials_data_test[looper].power_filtered[7:], sixtrials_testPredict[looper]))
                             norm_mean_err_train.append(ppf.norm_mean_error(sixtrials_data_train[looper].power_filtered[7:], sixtrials_trainPredict[looper]))
                             norm_mean_err_test.append(ppf.norm_mean_error(sixtrials_data_test[looper].power_filtered[7:], sixtrials_testPredict[looper]))
                     if 0:
                         squared_error = ppf.plot_squared_error_over_time(sixtrials_data_test[0].power_filtered[7:], sixtrials_testPredict[0])
                         plt.figure()
                         plt.plot(squared_error)
                         
                     all_r2_train_mean.append(np.mean(sixtrials_r2_train))
                     all_r2_test_mean.append(np.mean(sixtrials_r2_test))
                     all_r2_alternative_train_mean.append(np.mean(r2_alternative_train))
                     all_r2_alternative_test_mean.append(np.mean(r2_alternative_test))
                     all_rmse_train_mean.append(np.mean(sixtrials_rmse_train))
                     all_rmse_test_mean.append(np.mean(sixtrials_rmse_test))
                     all_nrmse_train_mean.append(np.mean(sixtrials_nrmse_train))
                     all_nrmse_test_mean.append(np.mean(sixtrials_nrmse_test))
                     all_norm_mean_err_train_mean.append(np.mean(norm_mean_err_train))
                     all_norm_mean_err_test_mean.append(np.mean(norm_mean_err_test))
                     
                     
                     all_r2_train_std.append(np.std(sixtrials_r2_train))
                     all_r2_test_std.append(np.std(sixtrials_r2_test))
                     all_r2_alternative_train_std.append(np.std(r2_alternative_train))
                     all_r2_alternative_test_std.append(np.std(r2_alternative_test))
                     all_rmse_train_std.append(np.std(sixtrials_rmse_train))
                     all_rmse_test_std.append(np.std(sixtrials_rmse_test))
                     all_nrmse_train_std.append(np.std(sixtrials_nrmse_train))
                     all_nrmse_test_std.append(np.std(sixtrials_nrmse_test))
                     all_norm_mean_err_train_std.append(np.std(norm_mean_err_train))
                     all_norm_mean_err_test_std.append(np.std(norm_mean_err_test))
                     
                     all_loss.append(sixtrials_loss)
                     #all_val_loss.append(sixtrials_val_loss)
                     all_belonging_name.append(str(subject_id) + '_' + str(trial_id) + '_' + model)
                 
                    
                     trial_r2_train_mean.append(np.mean(sixtrials_r2_train))
                     trial_r2_test_mean.append(np.mean(sixtrials_r2_test))
                     trial_r2_alternative_train_mean.append(np.mean(r2_alternative_train))
                     trial_r2_alternative_test_mean.append(np.mean(r2_alternative_test))
                     trial_rmse_train_mean.append(np.mean(sixtrials_rmse_train))
                     trial_rmse_test_mean.append(np.mean(sixtrials_rmse_test))
                     trial_nrmse_train_mean.append(np.mean(sixtrials_nrmse_train))
                     trial_nrmse_test_mean.append(np.mean(sixtrials_nrmse_test))
                     trial_norm_mean_err_train_mean.append(np.mean(norm_mean_err_train))
                     trial_norm_mean_err_test_mean.append(np.mean(norm_mean_err_test))
                     
                     trial_r2_train_std.append(np.std(sixtrials_r2_train))
                     trial_r2_test_std.append(np.std(sixtrials_r2_test))
                     trial_r2_alternative_train_std.append(np.std(r2_alternative_train))
                     trial_r2_alternative_test_std.append(np.std(r2_alternative_test))
                     trial_rmse_train_std.append(np.std(sixtrials_rmse_train))
                     trial_rmse_test_std.append(np.std(sixtrials_rmse_test))
                     trial_nrmse_train_std.append(np.std(sixtrials_nrmse_train))
                     trial_nrmse_test_std.append(np.std(sixtrials_nrmse_test))
                     trial_norm_mean_err_train_std.append(np.std(norm_mean_err_train))
                     trial_norm_mean_err_test_std.append(np.std(norm_mean_err_test))
                
                if plot_flag:
                    for m in range(0, len(sixtrials_data_test)):
                        plt.figure()
                        plt.title('Predicted Power vs Actual Power neural network_' + (str(subject_id) + '_' + str(trial_id) + '_' + model))
                        predicted_power = sixtrials_testPredict[m].reshape(len(sixtrials_testPredict[m]))
                        actual_power = sixtrials_data_test[m].power_filtered.reset_index(drop = True)
                        predicted_power = plt.plot(predicted_power, label = 'predicted_power', color = 'b')
                        actual_power = plt.plot(actual_power, label = 'measured_power', color = 'r')
                        plt.legend()
            
            subject_r2_train_mean.append(np.mean(trial_r2_train_mean))
            subject_r2_test_mean.append(np.mean(trial_r2_test_mean))
            subject_r2_alternative_train_mean.append(np.mean(trial_r2_alternative_train_mean))
            subject_r2_alternative_test_mean.append(np.mean(trial_r2_alternative_test_mean))
            subject_rmse_train_mean.append(np.mean(trial_rmse_train_mean))
            subject_rmse_test_mean.append(np.mean(trial_rmse_test_mean))
            subject_nrmse_train_mean.append(np.mean(trial_nrmse_train_mean))
            subject_nrmse_test_mean.append(np.mean(trial_nrmse_test_mean))
            subject_norm_mean_err_train_mean.append(np.mean(trial_norm_mean_err_train_mean))
            subject_norm_mean_err_test_mean.append(np.mean(trial_norm_mean_err_test_mean))
            
            subject_r2_train_std.append(np.mean(trial_r2_train_std))
            subject_r2_test_std.append(np.mean(trial_r2_test_std))
            subject_r2_alternative_train_std.append(np.mean(trial_r2_alternative_train_std))
            subject_r2_alternative_test_std.append(np.mean(trial_r2_alternative_test_std))
            subject_rmse_train_std.append(np.mean(trial_rmse_train_std))
            subject_rmse_test_std.append(np.mean(trial_rmse_test_std))
            subject_nrmse_train_std.append(np.mean(trial_nrmse_train_std))
            subject_nrmse_test_std.append(np.mean(trial_nrmse_test_std))
            subject_norm_mean_err_train_std.append(np.mean(trial_norm_mean_err_train_std))
            subject_norm_mean_err_test_std.append(np.mean(trial_norm_mean_err_test_std))
            
        tested_models_r2_train_mean.append(np.mean(subject_r2_train_mean))
        tested_models_r2_test_mean.append(np.mean(subject_r2_test_mean))
        tested_models_r2_alternative_train_mean.append(np.mean(subject_r2_alternative_train_mean))
        tested_models_r2_alternative_test_mean.append(np.mean(subject_r2_alternative_test_mean))
        tested_models_rmse_train_mean.append(np.mean(subject_rmse_train_mean))
        tested_models_rmse_test_mean.append(np.mean(subject_rmse_test_mean))
        tested_models_nrmse_train_mean.append(np.mean(subject_nrmse_train_mean))
        tested_models_nrmse_test_mean.append(np.mean(subject_nrmse_test_mean))
        tested_models_norm_mean_err_train_mean.append(np.mean(subject_norm_mean_err_train_mean))
        tested_models_norm_mean_err_test_mean.append(np.mean(subject_norm_mean_err_test_mean))
        
        tested_models_r2_train_std.append(np.mean(subject_r2_train_std))
        tested_models_r2_test_std.append(np.mean(subject_r2_test_std))
        tested_models_r2_alternative_train_std.append(np.mean(subject_r2_alternative_train_std))
        tested_models_r2_alternative_test_std.append(np.mean(subject_r2_alternative_test_std))
        tested_models_rmse_train_std.append(np.mean(subject_rmse_train_std))
        tested_models_rmse_test_std.append(np.mean(subject_rmse_test_std))
        tested_models_nrmse_train_std.append(np.mean(subject_nrmse_train_std))
        tested_models_nrmse_test_std.append(np.mean(subject_nrmse_test_std))
        tested_models_norm_mean_err_train_std.append(np.mean(subject_norm_mean_err_train_std))
        tested_models_norm_mean_err_test_std.append(np.mean(subject_norm_mean_err_test_std))
        
   
    
    
    if save_flag:
        data = pd.DataFrame({'models': tested_models, 'r2_alternative_test_mean': tested_models_r2_alternative_test_mean, 'r2_alternative_test_std': tested_models_r2_alternative_test_std, 'r2_alternative_train_mean': tested_models_r2_alternative_train_mean, 'r2_alternative_train_std': tested_models_r2_alternative_train_std, 'nrmse_test_mean': tested_models_nrmse_test_mean, 'nrmse_test_std': tested_models_nrmse_test_std,  'nrmse_train_mean': tested_models_nrmse_train_mean, 'nrmse_train_std': tested_models_nrmse_train_std, 'norm_mean_err_test_mean': tested_models_norm_mean_err_test_mean, 'norm_mean_err_test_std': tested_models_norm_mean_err_test_std, 'norm_mean_err_train_mean': tested_models_norm_mean_err_train_mean, 'norm_mean_err_train_std': tested_models_norm_mean_err_train_std})

        filename = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData' +   \
    '/OpenLoop/MatVsPython/Individual/Results_python_bayesian.csv'
        data.to_csv(filename)
        
    if single_model_save_flag:
        data_per_subject = pd.DataFrame({'subject': subject, 'r2_alternative_test_mean': subject_r2_alternative_test_mean, 'r2_alternative_test_std': subject_r2_alternative_test_std,  'r2_alternative_train_mean': subject_r2_alternative_train_mean, 'r2_alternative_train_std': subject_r2_alternative_train_std, 'nrmse_test_mean': subject_nrmse_test_mean, 'nrmse_test_std': subject_nrmse_test_std,  'nrmse_train_mean': subject_nrmse_train_mean, 'nrmse_train_std': subject_nrmse_train_std, 'r2_test_mean': subject_r2_test_mean, 'r2_test_std': subject_r2_test_std,  'r2_train_mean': subject_r2_train_mean, 'r2_train_std': subject_r2_train_std, 'norm_mean_err_test_mean': subject_norm_mean_err_test_mean, 'norm_mean_err_test_std': subject_norm_mean_err_test_std,  'norm_mean_err_train_mean': subject_norm_mean_err_train_mean, 'norm_mean_err_train_std': subject_norm_mean_err_train_std})
        filename = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData' +   \
    '/OpenLoop/MatVsPython/Individual/Results_python.csv'
        data_per_subject.to_csv(filename)
                
                
    

if which_model == 'bayesiansearch':
    # try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
    fivetrials_name = list()
    
    
    
    
    """load the data"""
    "changeable variables sub 1"
    
    for subject_id in subject:
        for trial_id in trials:
            load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/' + str(trial_id) + '/' + bayesiansearch_model_name + '.pkl'
            
            with open(load_path, 'rb') as f:
                onetrials_name = pickle.load(f)  
            fivetrials_name = np.append(fivetrials_name, onetrials_name) 
    print(fivetrials_name)