#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:52:38 2021

@author: patrickmayerhofer

PredictPower loads a dataset and runs an RNN on the 
input data to predict the power.
Specify filename and which variables you'd like to use as X and Y variable(s)'
"""

# libraries
import numpy as np
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
subject = range(3,9)
trial = range(1,4)
normalize_flag = 1
save_flag = 1
plot_flag = 1
which_model = 'custom' # 'dynamic_model', 'sentdex', 'custom'
# batch size: calculated based on lenght of trial

previous_flag = 0
X_variables = ['rpm_filtered', 'rpmdot_filtered']
Y_variable = ['power_filtered']

epochs = 1000
max_trials = 10
patience = 2000
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


for subject_id in subject:
    for trial_id in trial:
        "load csv files"
        filename = 'Subject' + str(subject_id) + '_' + str(trial_id)
        dir_load_file_train = dir_root + 'OpenLoop/CleanedCSV/' + filename + '_train.csv'
        dir_load_file_test = dir_root + 'OpenLoop/CleanedCSV/' + filename + '_test.csv'
        
        
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(name)
        dir_log = dir_root + '/Results/RNN/' + name
        
        "learning algorithm"
        # callbacks
        # Early Stopping
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            #min_delta=min_delta,
            patience=patience,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        
        """
        # Tensorboard
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=dir_log, histogram_freq=0, write_graph=True,
            write_images=False, update_freq='epoch', profile_batch=2,
            embeddings_freq=0, embeddings_metadata=None
        )
        """
        
        cb_list  = [es] # es, tb
        
        
        # load data
        data_train = pd.read_csv(dir_load_file_train)
        data_test = pd.read_csv(dir_load_file_test)
        
        # normalize?
        if normalize_flag:
            data_train.power_filtered = (data_train.power_filtered-data_train.power_filtered.min())/(data_train.power_filtered.max()-data_train.power_filtered.min())
            data_test.power_filtered = (data_test.power_filtered-data_test.power_filtered.min())/(data_test.power_filtered.max()-data_test.power_filtered.min())                             
        
        
        # if wanted,add another column with the previous power here
        if previous_flag:
            data_train = ppf.get_previous(data_train, previous_variable)
            data_test = ppf.get_previous(data_test, previous_variable)
        
        
        # prepare data, get reshaped training and test variables
        trainX, trainY = ppf.prepare_dataset(data_train, X_variables, Y_variable)
        testX, testY = ppf.prepare_dataset(data_test, X_variables, Y_variable)
        
        # batch_size 
        batch_size = len(trainX)
        
        if which_model == 'sentdex':
            #create model
            tuner = kerastuner.RandomSearch(
                    ppf.build_model_sentdex_tuner_cycling,
                    objective='val_loss',
                    max_trials=max_trials,  # how many model variations to test?
                    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
                    directory=dir_log,
                    )
            
            print("Prepared the model, starting to fit now. ")    
            # train all the models
            tuner.search(trainX,
                         trainY,
                         verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
                         epochs=epochs,
                         callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                         validation_data = (testX, testY),
                         batch_size = batch_size,
                         )
                
            #pickle.dump(tuner, open( dir_log + ".p", "wb" ) ) 
         
        # this will load a tuner pickle, and recreate the most successful model    
        elif which_model == 'dynamic_model': 
          dir_load_tuner = dir_root + 'Results/RNN/' + tuner_name
          tuner = pickle.load( open( dir_load_tuner + ".p", "rb" ))
          best_model = tuner.get_best_hyperparameters()[0].values
          model= ppf.build_model_dynamic(trainX, best_model)
        elif which_model == 'custom':
          print('1')
          model= ppf.create_model(trainX)
        
        
        if which_model != 'sentdex':
          print('2')
          history = model.fit(trainX, trainY,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle=False,
                                 batch_size = batch_size,
                                 callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                                 validation_data = (testX, testY))
          
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        if plot_flag == 1:
            plt.figure()
            plt.title('Predicted Power vs Actual Power normalized-test')
            predicted_power = plt.plot(testPredict[:,0], label = 'predicted_power', color = 'r')
            actual_power = plt.plot(data_test.power_filtered, label = 'measured_power', color = 'b')
            plt.legend()
            if save_flag:
                filename_test = dir_root + 'Graphs/Subject' + str(subject_id) + '_' + str(trial_id) + '_test.png'
                plt.savefig(filename_test, dpi=300)
            
            plt.figure()
            plt.title('Predicted Power vs Actual Power normalized-train')
            predicted_power = plt.plot(trainPredict[:,0], label = 'predicted_power', color = 'r')
            actual_power = plt.plot(data_train.power_filtered, label = 'measured_power', color = 'b')
            plt.legend()
            if save_flag:
                filename_train = dir_root + 'Graphs/Subject' + str(subject_id) + '_' + str(trial_id) + '_train.png'
                plt.savefig(filename_train, dpi=300)
        
        # evaluate the keras model
        r2_train = r2_score(data_train.power_filtered, trainPredict[:,0])
        r2_test = r2_score(data_test.power_filtered, testPredict[:,0])
        rmse_train = math.sqrt(mean_squared_error(data_train.power_filtered, trainPredict[:,0]))
        rmse_test = math.sqrt(mean_squared_error(data_test.power_filtered, testPredict[:,0]))
            
        "fill summary file and if true, save data to excel sheet"
        local_summary_file = ppf.save_data(local_summary_file, subject_id, trial_id, r2_train,r2_test,rmse_train,rmse_test);

# if in a loop, do this only in the very end
if save_flag:
    local_summary_file.to_excel(dir_summary_file)

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
