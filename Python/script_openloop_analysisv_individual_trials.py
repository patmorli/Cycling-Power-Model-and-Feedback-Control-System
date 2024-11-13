#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:08:50 2021

@author: patrickmayerhofer
"""

def script_openloop_analysisv_individual_trials(subject, trial, save_flag, which_model, tuner_epochs, epochs, max_trials, patience, min_delta, dir_root, oa, ppf, range_, layers_nodes, learning_rate, normalize_flag, save_name, X_variables, Y_variable, window_size, step):
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
    import os
    
    "changeable variables"
    #subject = range(11)
    # try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
    
   
    
    #filename = 'Subject4_2'
    
    "load summary file"
    # excel summary file
    dir_summary_file = dir_root + 'Summary.xlsx'
    summary_file = pd.read_excel(dir_summary_file)
    local_summary_file = summary_file # to change variables in the summary_file and save it later
    
    "learning algorithm"
    # callbacks
    # Early Stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor="mse",
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
           
            if which_model == "custom":
                layer = [0] # just for the function input, not needed in 'custom'
                node = [0]  # just for the function input, not needed in 'custom'
                fivetrials_r2_train, fivetrials_r2_test,                \
                       fivetrials_rmse_train, fivetrials_rmse_test,            \
                       fivetrials_nrmse_train, fivetrials_nrmse_test,           \
                       fivetrials_data_train, fivetrials_data_test,            \
                       fivetrials_trainPredict, fivetrials_testPredict,         \
                       fivetrials_loss =       \
                       oa.custom_algorithm_individual(which_model, subject_id, trial_id,   \
                       layer, node, increments, data, X_variables, Y_variable, \
                       epochs, cb_list, range_,         \
                       layers_nodes, learning_rate, window_size, step) 
            
           
            
            elif which_model == 'hyperband' or which_model == 'bayesiansearch' or which_model == 'randomsearch':
                """load and organize data"""   
                # it should be fivetrials
                fivetrials_name = oa.kerastuneralgorithm_individual(data, X_variables, Y_variable, increments, epochs, dir_save, cb_list, which_model,subject_id, max_trials, save_flag, save_name, window_size, step, range_)
                        
                        # when trial of subject is done, save it                
            if save_flag:
                print('Hopped into saving mode')
                if which_model == 'custom':
                    if os.path.isdir(dir_save) != True:
                        os.makedirs(dir_save)
                    dir_save = dir_save + str(layers_nodes) + str(learning_rate) + '/'
                    if os.path.isdir(dir_save) != True:
                        os.makedirs(dir_save)
                    save_path = dir_save + 'summary_results.pkl'
                    print(save_path)
                    with open(save_path, 'wb') as f:
                        pickle.dump([fivetrials_r2_train, fivetrials_r2_test, fivetrials_rmse_train, fivetrials_rmse_test, fivetrials_nrmse_train, fivetrials_nrmse_test, fivetrials_data_train, fivetrials_data_test, fivetrials_trainPredict, fivetrials_testPredict, fivetrials_loss], f)  
                    print('should be saved now ')
           
                elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
                    print('should be saving now') 
                    save_path = dir_save + save_name + '.pkl'
                    print(save_path)
                    
                    with open(save_path, 'wb') as f:
                        pickle.dump([fivetrials_name], f)  
                    print('should be saved now ')
    """       
    return fivetrials_r2_train, fivetrials_r2_test, fivetrials_rmse_train,     \
        fivetrials_rmse_test, fivetrials_nrmse_train, fivetrials_nrmse_test,    \
        fivetrials_data_train, fivetrials_data_test,     \
        fivetrials_trainPredict, fivetrials_testPredict, fivetrials_loss 
    """            
                
            