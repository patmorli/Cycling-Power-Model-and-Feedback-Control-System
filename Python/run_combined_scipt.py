#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:23:41 2021

@author: patrickmayerhofer
"""

import script_openloop_analysisv_combined_trials_v2 as script
import PredictPower_functions as ppf
import openloop_algorithms as oa

subject = [9] #9,11,12 are pilot subjects
trial_id = [1,2]
save_flag = 0
which_model = 'bayesiansearch' # 'hyperband', 'forloop', 'randomsearch', 'bayesiansearch', 'custom'

save_name = '2_32_1024_32'

X_variables = ['rpm_filtered','rpmdot_filtered', 'gear_ratio'] #'rpmdot_filtered'
Y_variable = ['power_filtered']

# creates windows and saves in a dataframe --> put this outside of this function
step = 1
window_size = 8

layers = [1] #[1,2,3,4]s
nodes = [8] #[8,16,32]
layers_nodes = [8] # this arrray will be the number of nodes for each layer. more numbers -> more layers
learning_rate = 0.01

range_ = [0,6] #[0,5] # default is [0,5]. If there was problem in calculation, this can be adjusted so that we don't have to do full subject again. 


tuner_epochs = 10
epochs = 1000
max_trials = 5
patience = 300
min_delta = 0.1

normalize_flag = 0

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/'

sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train,     \
                   sixtrials_rmse_test, sixtrials_nrmse_train,                      \
                   sixtrials_nrmse_test, sixtrials_data_train,                      \
                   sixtrials_data_test, sixtrials_trainPredict,                     \
                   sixtrials_testPredict =                      \
                   script.script_openloop_analysisv_combined_trials(subject, trial_id, save_flag, which_model, tuner_epochs, epochs, max_trials, layers, nodes, patience, min_delta, dir_root, oa, ppf, range_, layers_nodes, learning_rate, normalize_flag, save_name, X_variables, Y_variable, window_size, step)
