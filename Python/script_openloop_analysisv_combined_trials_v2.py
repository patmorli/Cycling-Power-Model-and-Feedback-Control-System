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
def script_openloop_analysisv_combined_trials(subject, trial_id, save_flag, which_model, tuner_epochs, epochs, max_trials, layers, nodes, patience, min_delta, dir_root, oa, ppf,range_, layers_nodes, learning_rate, normalize_flag, save_name, X_variables, Y_variable, window_size, step):
    # libraries
    import pandas as pd
    import sys 
    sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
    import tensorflow as tf
    import pickle
    import kerastuner
    from kerastuner.engine.hyperparameters import HyperParameters
    from numpy import zeros, ones
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    import math
    import os
    
    
    "changeable variables"
    #subject = range(11)
    # try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
    
    # batch size: calculated based on lenght of trial
    """
    hp = HyperParameters()
    hp.Choice('learning_rate', [0.001, 0.01])
    hp.Int('hidden_layers', 1, 5, step = 1, default = 1)
    hp.Int('units', 8, 64, step = 8, default = 8)
    """
    
    
    
    fbracket = 39
    
    
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
        dir_save = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Combined/'  \
        + which_model + '/Subject' + str(subject_id) + '/'
        """load and organize data"""
        increments, data = ppf.load_and_organize_data_combined(subject_id, trial_id, \
                      dir_root, normalize_flag, summary_file, fbracket)
          
            
        if which_model == 'custom':
            # save for different nodes and layers and subjects           
            layer = [0] # just for the function input, not needed in 'custom'
            node = [0]  # just for the function input, not needed in 'custom'
            """run full algorithm, not best function, but it gives better overview"""
            sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train,     \
            sixtrials_rmse_test, sixtrials_nrmse_train,                      \
            sixtrials_nrmse_test, sixtrials_data_train,                      \
            sixtrials_data_test, sixtrials_trainPredict,                     \
            sixtrials_testPredict, sixtrials_loss       \
            = oa.custom_algorithm_combined(which_model, subject_id, trial_id, layer,   \
                                           node, increments, data,           \
                                           X_variables, Y_variable, epochs,  \
                                           range_, cb_list, layers_nodes,            \
                                           learning_rate, window_size, step)
            
        
        elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
            """load and organize data"""
            
            "run hyperband for all 6 increments"
            sixtrials_name = oa.kerastuneralgorithm_combined(           \
               data, X_variables, Y_variable, increments,               \
               tuner_epochs, dir_save, cb_list, which_model, subject_id, max_trials, save_flag, save_name, window_size, step, range_)
            print('Done training tuner')
            
            
                        
        if save_flag:
            print('Hopped into saving mode')
           
            if which_model == 'custom' or which_model == 'custom_new':
                if os.path.isdir(dir_save) != True:
                    os.makedirs(dir_save)
                dir_save = dir_save + str(layers_nodes) + str(learning_rate) + '/'
                if os.path.isdir(dir_save) != True:
                    os.makedirs(dir_save)
                save_path = dir_save + 'summary_results.pkl'
                print(save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump([sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train, sixtrials_rmse_test, sixtrials_nrmse_train, sixtrials_nrmse_test, sixtrials_data_train, sixtrials_data_test, sixtrials_trainPredict, sixtrials_testPredict, sixtrials_loss], f)  
                print('should be saved now ')
                
                      
                
           
            elif which_model == 'hyperband' or 'bayesiansearch' or 'randomsearch':
                print('should be saving now')
                #save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Hyperband/CombinedSubject' + str(subject_id) + '_' + '.pkl'
                save_path = dir_save + save_name + '.pkl'
                print(save_path)
                
                with open(save_path, 'wb') as f:
                    pickle.dump([sixtrials_name], f)  
                print('should be saved now ')
    """ 
    return sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train,     \
                   sixtrials_rmse_test, sixtrials_nrmse_train,                      \
                   sixtrials_nrmse_test, sixtrials_data_train,                      \
                   sixtrials_data_test, sixtrials_trainPredict,                     \
                   sixtrials_testPredict
    """ 
    
                    