#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:34:16 2021

@author: patrickmayerhofer
"""

import script_openloop_analysisv_individual_trials as script
import PredictPower_functions as ppf
import openloop_algorithms as oa

subject = [9]
trial = [1]
save_flag = 0
which_model = 'custom' # 'custom', 'hyperband', 'forloop', 'randomsearch', 'bayesiansearch'
save_name = 'yes' # for kerastuner stuff

X_variables = ['rpm_filtered','rpmdot_filtered'] #'rpmdot_filtered'
Y_variable = ['power_filtered']

# creates windows and saves in a dataframe --> put this outside of this function
step = 1
window_size = 8

layers = [1] #[1,2,3,4]
nodes = [8] #[8,16,32]

normalize_flag = 0

layers_nodes = [8] # this arrray will be the number of nodes for each layer. more numbers -> more layers
learning_rate = 0.01

range_ = [0,3] #[0,3] # default is [0,3]. If there was problem in calculation, this can be adjusted so that we don't have to do full subject again. 

tuner_epochs = 50 #5000
epochs = 10
max_trials = 300
patience = 100
min_delta = 0.01

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/'

fivetrials_r2_train, fivetrials_r2_test,                \
                fivetrials_rmse_train, fivetrials_rmse_test,            \
                fivetrials_nrmse_train, fivetrials_nrmse_test,          \
                fivetrials_data_train, fivetrials_data_test,            \
                fivetrials_trainPredict, fivetrials_testPredict,        \
                fivetrials_loss =       \
                script.script_openloop_analysisv_individual_trials(subject, trial, save_flag, which_model, tuner_epochs, epochs, max_trials, patience, min_delta, dir_root, oa, ppf, range_, layers_nodes, learning_rate, normalize_flag, save_name, X_variables, Y_variable, window_size, step)
