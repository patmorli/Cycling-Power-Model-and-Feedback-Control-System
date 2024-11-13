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
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import kerastuner
import pickle

"changeable variables"
#subject = range(11)
# try subjects and trials: 1_1, 1_3, 8_1, 8_2, 11_1-3, 12_1-2
subject = [8]
trial_id = [1,2]
normalize_flag = 1
save_flag = 0
which_model = 'sentdex' # 'dynamic_model', 'sentdex', 'custom'
# batch size: calculated based on lenght of trial

previous_flag = 0
X_variables = ['rpm_filtered', 'rpmdot_filtered', 'gear_ratio']
Y_variable = ['power_filtered']

epochs = 10000
max_trials = 100
patience = 100
min_delta = 0.0001

tuner_name = '20210505-195609' # if we want to use a specific tuner that has been developed earlier
previous_variable = Y_variable
fbracket = 39

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
    monitor="loss",
    min_delta=min_delta,
    patience=patience,
    verbose=1, #will show progress bar
    mode="auto",
    baseline=None,
    restore_best_weights=True, #restores the weights of the best epoch not the last one (false)
)

cb_list  = [es] # es, tb

for subject_id in subject:
        "load csv files"
        filename1 = 'Subject' + str(subject_id) + '_' + str(trial_id[0])
        dir_load_file1 = dir_root + 'OpenLoop/CleanedCSV/' + filename1 + '.csv'
        data1 = pd.read_csv(dir_load_file1)
    
        filename2 = 'Subject' + str(subject_id) + '_' + str(trial_id[1])
        dir_load_file2 = dir_root + 'OpenLoop/CleanedCSV/' + filename2 + '.csv'
        data2 = pd.read_csv(dir_load_file2)
        
        
        if normalize_flag:
            data1.power_filtered = (data1.power_filtered-data1.power_filtered.min())/(data1.power_filtered.max()-data1.power_filtered.min())                       
            data2.power_filtered = (data2.power_filtered-data2.power_filtered.min())/(data2.power_filtered.max()-data2.power_filtered.min())                       
       
        """get gear ratio, add to data and concatenate data"""
        rbracket_1 = summary_file.where(summary_file.ID == subject_id).dropna(how='all').rbracket_1
        gr1 = fbracket/rbracket_1;
        data1.insert(2, "gear_ratio", float(gr1))
        rbracket_2 = summary_file.where(summary_file.ID == subject_id).dropna(how='all').rbracket_2
        gr2 = fbracket/rbracket_2;
        data2.insert(2, "gear_ratio", float(gr2))
        
        
        """create indizes for 3(?) same-sized test sets from each trial"""
        # create 3 test sets per trial
        l1_ = len(data1)
        l1_3 = round(l1_/3)
        increments1 = (0, l1_3,l1_3*2, l1_)
        
        l2_ = len(data2)
        l2_3 = round(l2_/3)
        increments2 = (0, l2_3,l2_3*2, l2_)
        
        # merge data1 and data2 and increments
        data = [data1, data2]
        data = pd.concat(data, ignore_index = True)
        increments2 = tuple([i + l1_ for i in increments2])
        increments = increments1 + increments2[1:len(increments2)]
        increments = list(increments)
        for i in range(len(increments)-1):
            increments[i+1] = increments[i+1] +1
        
        layers = [1] #[1,2,3,4]
        nodes = [8] #[8,16,32]
        
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
               
          
               # save within same layer and same node but 5 different trials
               fivetrials_r2_train = np.array([])
               fivetrials_r2_test = np.array([])
               fivetrials_rmse_train = np.array([])
               fivetrials_rmse_test = np.array([])
               fivetrials_data_train = list()
               fivetrials_data_test = list()
               fivetrials_trainPredict = list()
               fivetrials_testPredict = list()
               
               #do the test 5 times with about 16% test set and about 84% train set
               layers_nodes = [str(subject_id) + '_' + str(layer) + '_' + str(node)]
               for i in range(0,6):
                    "get variables for calculation and prepare data"
                    #create array that shows what should be used for test and training
                    ones_matrix_test = zeros(increments[6]-1)
                    ones_matrix_train = ones(increments[6]-1)
                    for u in range(increments[i],increments[i+1]-1):
                        ones_matrix_test[u] = 1;
                        ones_matrix_train[u] = 0;
                    
                    # divide in train and data set
                    data_test = data[ones_matrix_test>0]
                    data_train = data[ones_matrix_train>0]
                    
                    # prepare data, get reshaped training and test variables
                    trainX, trainY = ppf.prepare_dataset(data_train, X_variables, Y_variable)
                    testX, testY = ppf.prepare_dataset(data_test, X_variables, Y_variable)
                    
                    
                    # batch_size 
                    batch_size = len(trainX)
                    
                    """start the fun modeling"""
                    if which_model == 'custom':
                        model = ppf.forloop_model(trainX, layer, node)
                        history = model.fit(trainX, trainY,
                                     epochs=epochs,
                                     verbose=1,
                                     shuffle=False,
                                     batch_size = batch_size,
                                     callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                                     #validation_data = (testX, testY)
                                     )
                        
                        # make predictions
                        trainPredict = model.predict(trainX)
                        testPredict = model.predict(testX)
                    elif which_model == 'sentdex':
                        #create model
                      
                        tuner = kerastuner.Hyperband(
                                  ppf.build_model_sentdex_tuner_cycling,
                                  objective='val_loss',
                                  max_epochs=10000,  
                                  #directory=log_dir,
                                  )
                        
                        # train all the models
                        tuner.search(trainX,
                                    trainY,
                                    verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
                                    epochs=epochs,
                                    callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                                    validation_data = (testX, testY),
                                    batch_size = batch_size,
                                    )
                    
                    """
                    plt.figure()
                    plt.title('Predicted Power vs Actual Power normalized-test')
                    predicted_power = plt.plot(testPredict[:,0], label = 'predicted_power', color = 'r')
                    actual_power = plt.plot(data_test.power_filtered, label = 'measured_power', color = 'b')
                    plt.legend()
                   """
                    # evaluate the keras model
                    r2_train = r2_score(data_train.power_filtered, trainPredict[:,0])
                    r2_test = r2_score(data_test.power_filtered, testPredict[:,0])
                    rmse_train = math.sqrt(mean_squared_error(data_train.power_filtered, trainPredict[:,0]))
                    rmse_test = math.sqrt(mean_squared_error(data_test.power_filtered, testPredict[:,0]))
         
  
                    # store results of 5 different trials
                    fivetrials_r2_train = np.append(fivetrials_r2_train, r2_train)
                    fivetrials_r2_test = np.append(fivetrials_r2_test, r2_test)
                    fivetrials_rmse_train = np.append(fivetrials_rmse_train, rmse_train)
                    fivetrials_rmse_test = np.append(fivetrials_rmse_test, rmse_test)
                    fivetrials_data_train.append(data_train)
                    fivetrials_data_test.append(data_test)
                    fivetrials_trainPredict.append(trainPredict)
                    fivetrials_testPredict.append(testPredict)
                    
                
                    
               
              # save these results for respective layer and node number
               save_layers_nodes.append(layers_nodes)
               save_r2_train.append(fivetrials_r2_train)
               save_r2_test.append(fivetrials_r2_test)
               save_rmse_train.append(fivetrials_rmse_train)
               save_rmse_test.append(fivetrials_rmse_test)
               save_data_train.append(fivetrials_data_train)
               save_data_test.append(data_test)
               save_trainPredict.append(fivetrials_trainPredict)
               save_testPredict.append(fivetrials_testPredict)
                        
        if save_flag:
            save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/CombinedSubject' + str(subject_id) + '_' + '.pkl'
    
         
        with open(save_path, 'wb') as f:
            pickle.dump([save_r2_train, save_r2_test, 
                              save_rmse_train, save_rmse_test, 
                              save_data_train, save_data_test,
                              save_trainPredict, save_testPredict,
                              save_layers_nodes], f)    
        
        
       
        """  
        with open(save_path, 'rb') as f:
           save_r2_train, save_r2_test, save_rmse_train, save_rmse_test,save_data_train, save_data_test, save_trainPredict, save_testPredict  =  pickle.load(f)     
        """
        """Patrick: make excel file to save and later compare results of best combination of nodes and layers"""
          
          
                
      
        
        
            
        "fill summary file and if true, save data to excel sheet"
        #local_summary_file = ppf.save_data(local_summary_file, subject_id, trial_id, r2_train,r2_test,rmse_train,rmse_test);

# if in a loop, do this only in the very end
#if save_flag:
  #  local_summary_file.to_excel(dir_summary_file)

"""
#train model
history = model.fit(trainX, trainY, epochs=epochs, verbose=2, shuffle=False)




plt.figure()
plt.plot(testY)
plt.plot(testPredict)

# invert predictions
r2_train = r2_score(trainY, trainPredict)
r2_test = r2_score(testY, testPredict)
"""
                