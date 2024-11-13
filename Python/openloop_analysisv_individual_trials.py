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
subject = [1]
trial = [1,3]
normalize_flag = 1
save_flag = 0
which_model = 'bayesiansearch' # 'hyperband', 'forloop', 'randomsearch', 'bayesiansearch'
# batch size: calculated based on lenght of trial

previous_flag = 0
X_variables = ['rpm_filtered','rpmdot_filtered'] #'rpmdot_filtered'
Y_variable = ['power_filtered']

tuner_epochs = 25
epochs = 10000
max_trials = 10
patience = 1000
min_delta = 0.0001

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
        "load csv files"
        filename = 'Subject' + str(subject_id) + '_' + str(trial_id)
        dir_load_file = dir_root + 'OpenLoop/CleanedCSV/' + filename + '.csv'
        
        
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(name)
        dir_log = dir_root + '/ResultsPython/RNN/Individuals/' + name
        
        # load data
        data = pd.read_csv(dir_load_file)
        
        if normalize_flag:
            data.power_filtered = (data.power_filtered-data.power_filtered.min())/(data.power_filtered.max()-data.power_filtered.min())                       
       
        
        "put in 5 sets"
        l = len(data)
        l5 = round(l/5)
        increments = (0, l5, l5*2, l5*3, l5*4, l)
        data1 = data[increments[0]:increments[1]];
        data2 = data[increments[1]:increments[2]];
        data3 = data[increments[2]:increments[3]];
        data4 = data[increments[3]:increments[4]];
        data5 = data[increments[4]:increments[5]];
        
        """
        for i in range(0,5):
            "get variables for calculation and prepare data"
            #create array that shows what should be used for test and training
            ones_matrix_test = zeros(l)
            ones_matrix_train = ones(l)
            for u in range(increments[i],increments[i+1]):
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
            """
            
      
        layers = [1] #[1,2,3,4]
        nodes = [8, 16] #[8,16,32]
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
                fivetrials_layers_nodes = list()
                
                
                
                #do the test 5 times with 80% training and 20% test
                layers_nodes = [str(subject_id) + '_' + str(trial_id) + '_' + str(layer) + '_' + str(node)]
                for i in range(0,5):
                    "get variables for calculation and prepare data"
                    #create array that shows what should be used for test and training
                    ones_matrix_test = zeros(l)
                    ones_matrix_train = ones(l)
                    for u in range(increments[i],increments[i+1]):
                        ones_matrix_test[u] = 1;
                        ones_matrix_train[u] = 0;
                        
                    # divide in train and data set
                    data_test = data[ones_matrix_test>0]
                    data_train = data[ones_matrix_train>0]
                    
                    #[samples, time steps, features]
                    # prepare data, get reshaped training and test variables
                    trainX, trainY = ppf.prepare_dataset(data_train, X_variables, Y_variable)
                    testX, testY = ppf.prepare_dataset(data_test, X_variables, Y_variable)
                
                    # batch_size 
                    batch_size = len(trainX)
                    model = ppf.forloop_model(trainX, layer, node)
                    history = model.fit(trainX, trainY,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle=False,
                                 batch_size = batch_size,
                                 callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                                 validation_data = (testX, testY)
                                 )
                    
                    # make predictions
                    trainPredict = model.predict(trainX)
                    testPredict = model.predict(testX)
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
                save_data_test.append(fivetrials_data_test)
                save_trainPredict.append(fivetrials_trainPredict)
                save_testPredict.append(fivetrials_testPredict)
                """
                # make variables empty again
                fivetrials_r2_train = np.array([])
                fivetrials_r2_test = np.array([])
                fivetrials_rmse_train = np.array([])
                fivetrials_rmse_test = np.array([])
                fivetrials_data_train = list()
                fivetrials_data_test = list()
                fivetrials_trainPredict = list()
                fivetrials_testPredict = list()
                fivetrials_layers_nodes = list()
                """
                # when trial of subject is done, save it                
        if save_flag:
            save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Individuals/Subject' + str(subject_id) + '_' + str(trial_id) + '.pkl'
       
                
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
