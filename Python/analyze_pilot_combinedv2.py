#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:06:46 2021

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
subject=[7]
#subject = [9,11,12]
which_model = 'custom' # beayesiansearch, custom
bayesiansearch_model_name = '1_16_1024_16'
plot_flag = 1 # creates hundreds of plots when running through (not sure why yet) make breakpoint for now
save_flag = 0 
per_subject_save_flag = 1 # if only checking one single model and 

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/' +   \
    '/OpenLoop/ResultsPython/FindAlgorithm/Combined/'

if which_model == 'custom':
    
    tested_models_r2_train = list()
    tested_models_r2_test = list()
    tested_models_rmse_train = list()
    tested_models_rmse_test = list()
    tested_models_nrmse_train = list()
    tested_models_nrmse_test = list()
    tested_models_r2_alternative_train = list()
    tested_models_r2_alternative_test = list()
    tested_models_mean_norm_err_train = list()
    tested_models_mean_norm_err_test=list()
    
    #load_path = dir_root + which_model + '/Subject8/'
    #tested_models = [name for name in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, name))]
    

    for subject_id in subject:
        load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/'
        tested_models = [name for name in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, name))]
        tested_models = ['[8]0.01'] #if you want to replace
    #if 1:
        for i in range(0, len(tested_models)):
            #i = 1
            # prepare variables
            all_r2_train_mean = list()
            all_r2_test_mean = list()
            all_r2_alternative_train_mean = list()
            all_r2_alternative_test_mean = list()
            all_rmse_train_mean = list()
            all_rmse_test_mean = list()
            all_nrmse_train_mean = list()
            all_nrmse_test_mean = list()
            all_mean_norm_err_train_mean = list()
            all_mean_norm_err_test_mean = list()
            
            all_r2_train_std = list()
            all_r2_test_std = list()
            all_r2_alternative_train_std = list()
            all_r2_alternative_test_std = list()
            all_rmse_train_std = list()
            all_rmse_test_std = list()
            all_nrmse_train_std = list()
            all_nrmse_test_std = list()
            all_mean_norm_err_train_std = list()
            all_mean_norm_err_test_std = list()
            
            all_loss = list()
            all_val_loss = list() 
            all_belonging_name = list()
            
            for subject_id in subject:
                load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/'
               # without nrmse:
                """   
                with open(load_path + tested_models + '/summary_results.pkl', 'rb') as f:
                    sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train, sixtrials_rmse_test, sixtrials_data_train, sixtrials_data_test, sixtrials_trainPredict, sixtrials_testPredict, sixtrials_loss, sixtrials_val_loss = pickle.load(f)  
                    all_r2_train.append(np.mean(sixtrials_r2_train))
                    all_r2_test.append(np.mean(sixtrials_r2_test))
                    all_rmse_train.append(np.mean(sixtrials_rmse_train))
                    all_rmse_test.append(np.mean(sixtrials_rmse_test))
                    all_loss.append(sixtrials_loss)
                    all_val_loss.append(sixtrials_val_loss)
                    all_belonging_name.append(str(subject_id) + '_' + tested_models[i])
                """
               #with nrmse:
                
                with open(load_path + tested_models[i] + '/summary_results.pkl', 'rb') as f:
                     sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train, sixtrials_rmse_test, sixtrials_nrmse_train, sixtrials_nrmse_test, sixtrials_data_train, sixtrials_data_test, sixtrials_trainPredict, sixtrials_testPredict, sixtrials_loss = pickle.load(f)  
                     
                     if 1:
                         r2_alternative_train = list()
                         r2_alternative_test = list()
                         mean_norm_err_train = list()
                         mean_norm_err_test = list()
                         for looper in range(len(sixtrials_data_test)):
                             r2_alternative_train.append(ppf.r2_alternative(sixtrials_data_train[looper].power_filtered[7:], sixtrials_trainPredict[looper]))
                             r2_alternative_test.append(ppf.r2_alternative(sixtrials_data_test[looper].power_filtered[7:], sixtrials_testPredict[looper]))
                             mean_norm_err_train.append(ppf.norm_mean_error(sixtrials_data_train[looper].power_filtered[7:], sixtrials_trainPredict[looper]))
                             mean_norm_err_test.append(ppf.norm_mean_error(sixtrials_data_test[looper].power_filtered[7:], sixtrials_testPredict[looper]))
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
                     all_mean_norm_err_train_mean.append(np.mean(mean_norm_err_train))
                     all_mean_norm_err_test_mean.append(np.mean(mean_norm_err_test))
                     
                     
                     all_r2_train_std.append(np.std(sixtrials_r2_train))
                     all_r2_test_std.append(np.std(sixtrials_r2_test))
                     all_r2_alternative_train_std.append(np.std(r2_alternative_train))
                     all_r2_alternative_test_std.append(np.std(r2_alternative_test))
                     all_rmse_train_std.append(np.std(sixtrials_rmse_train))
                     all_rmse_test_std.append(np.std(sixtrials_rmse_test))
                     all_nrmse_train_std.append(np.std(sixtrials_nrmse_train))
                     all_nrmse_test_std.append(np.std(sixtrials_nrmse_test))
                     all_mean_norm_err_train_std.append(np.std(mean_norm_err_train))
                     all_mean_norm_err_test_std.append(np.std(mean_norm_err_test))
                     
                     all_loss.append(sixtrials_loss)
                     #all_val_loss.append(sixtrials_val_loss)
                     all_belonging_name.append(str(subject_id) + '_' + tested_models[i])
                 
                
                if plot_flag:
                    for m in range(0, len(sixtrials_data_test)):
                        plt.figure()
                        plt.title('Predicted Power vs Actual Power neural network')
                        predicted_power = sixtrials_testPredict[m].reshape(len(sixtrials_testPredict[m]))
                        actual_power = sixtrials_data_test[m].power_filtered.reset_index(drop = True)
            
                        predicted_power = plt.plot(predicted_power, label = 'predicted_power', color = 'b')
                        actual_power = plt.plot(actual_power, label = 'measured_power', color = 'r')
                        plt.legend()
            
            
            tested_models_r2_train.append(np.mean(all_r2_train_mean))
            tested_models_r2_test.append(np.mean(all_r2_test_mean))
            tested_models_r2_alternative_train.append(np.mean(all_r2_alternative_train_mean))
            tested_models_r2_alternative_test.append(np.mean(all_r2_alternative_test_mean))
            tested_models_rmse_train.append(np.mean(all_rmse_train_mean))
            tested_models_rmse_test.append(np.mean(all_rmse_test_mean))
            tested_models_nrmse_train.append(np.mean(all_nrmse_train_mean))
            tested_models_nrmse_test.append(np.mean(all_nrmse_test_mean))
            tested_models_mean_norm_err_train.append(np.mean(all_mean_norm_err_train_mean))
            tested_models_mean_norm_err_test.append(np.mean(all_mean_norm_err_test_mean))
    
    
    if per_subject_save_flag:
        data_per_subject = pd.DataFrame({'subject': subject,'r2_alternative_test_mean': all_r2_alternative_test_mean, 'r2_alternative_test_std': all_r2_alternative_test_std, 'r2_alternative_train_mean': all_r2_alternative_train_mean, '2_alternative_train_std': all_r2_alternative_train_std, 'nrmse_test_mean': all_nrmse_test_mean, 'nrmse_test_std': all_nrmse_test_std, 'nrmse_train_mean': all_nrmse_train_mean, 'nrmse_train_std': all_nrmse_train_std, 'r2_test_mean': all_r2_test_mean, 'r2_test_std': all_r2_test_std, 'r2_train_mean': all_r2_train_mean, 'r2_train_std': all_r2_train_std, 'rmse_test_mean': all_rmse_test_mean, 'rmse_test_std': all_rmse_test_std, 'rmse_train_mean': all_rmse_train_mean, 'rmse_train_std': all_rmse_train_std, 'mean_norm_err_test_mean': all_mean_norm_err_test_mean, 'mean_norm_err_test_std': all_mean_norm_err_test_std, 'mean_norm_err_train_mean': all_mean_norm_err_train_mean, 'mean_norm_err_train_std': all_mean_norm_err_train_std})

        filename = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/' +   \
    '/OpenLoop/MatVsPython/Combined/Results_python.csv'
        data_per_subject.to_csv(filename)
        
    if save_flag:
        print('not implemented currently')
        
    
                
                
    

if which_model == 'bayesiansearch':
    # try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
    sixtrials_name = list()
    
    
    
    
    """load the data"""
    "changeable variables sub 1"
    
    for subject_id in subject:
        load_path = dir_root + which_model + '/Subject' + str(subject_id) + '/' + bayesiansearch_model_name + '.pkl'
        
        with open(load_path, 'rb') as f:
            onetrials_name = pickle.load(f)  
        sixtrials_name = np.append(sixtrials_name, onetrials_name) 
    print(sixtrials_name)
        
        
    