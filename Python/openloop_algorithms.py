#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:50:42 2021

@author: patrickmayerhofer
"""

import numpy as np
from numpy import zeros, ones
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
import PredictPower_functions as ppf
import kerastuner
import pickle
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from WindowGenerator import WindowGenerator


def custom_algorithm_combined(which_model, subject_id, trial_id, layer, node, increments, data, X_variables, Y_variable, epochs, range_,cb_list, layers_nodes, learning_rate, window_size, step) :
    
    variables = [X_variables + Y_variable][0]
    data = data[variables]   
    
    # save within same layer and same node but 6 different trials
    sixtrials_r2_train = np.array([])
    sixtrials_r2_test = np.array([])
    sixtrials_rmse_train = np.array([])
    sixtrials_rmse_test = np.array([])
    sixtrials_nrmse_train = np.array([])
    sixtrials_nrmse_test = np.array([])
    sixtrials_data_train = list()
    sixtrials_data_test = list()
    sixtrials_trainPredict = list()
    sixtrials_testPredict = list()
    sixtrials_loss = list()
    sixtrials_val_loss = list()
    
    
    for i in range(range_[0], range_[1]):
        print('--------------------------------------------------------------- trial:', str(i))
        "get variables for calculation and prepare data"
        #create array that shows what should be used for test and training
        dataset_train, dataset_test, data_train, data_test = ppf.create_train_test_tensor(data, window_size, step, X_variables, Y_variable, range_, increments, i)
        
        
        """start the fun modeling"""
        model = ppf.custom_model(window_size, len(X_variables), layers_nodes, learning_rate)
        history = model.fit(dataset_train,
                     epochs=epochs,
                     verbose=1,
                     shuffle=False,
                     callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                     #validation_split = 0.2,
                     #validation_data = (testX, testY)
                     )
            
       
        # make predictions
        trainPredict = model.predict(dataset_train)
        testPredict = model.predict(dataset_test)
        
  
    
        # evaluate the keras model
        r2_train, rmse_train, nrmse_train = ppf.cost(data_train[window_size-1:].power_filtered, trainPredict[:,0])
        r2_test, rmse_test, nrmse_test = ppf.cost(data_test[window_size-1:].power_filtered, testPredict[:,0])
        
        plt.figure()
        plt.title('Predicted Power vs Actual Power normalized-test')
        predicted_power = plt.plot(testPredict[:,0], label = 'predicted_power', color = 'r')
        actual_power = plt.plot(data_test[window_size-1:].power_filtered.reset_index(drop = True), label = 'measured_power', color = 'b')
        plt.legend()
        
        # store results of 6 different trials
        sixtrials_r2_train = np.append(sixtrials_r2_train, r2_train)
        sixtrials_r2_test = np.append(sixtrials_r2_test, r2_test)
        sixtrials_rmse_train = np.append(sixtrials_rmse_train, rmse_train)
        sixtrials_rmse_test = np.append(sixtrials_rmse_test, rmse_test)
        sixtrials_nrmse_train = np.append(sixtrials_nrmse_train, nrmse_train)
        sixtrials_nrmse_test = np.append(sixtrials_nrmse_test, nrmse_test)
        sixtrials_data_train.append(data_train)
        sixtrials_data_test.append(data_test)
        sixtrials_trainPredict.append(trainPredict)
        sixtrials_testPredict.append(testPredict)
        sixtrials_loss.append(history.history['loss'])
        #sixtrials_val_loss.append(history.history['val_loss'])
    
    return sixtrials_r2_train, sixtrials_r2_test, sixtrials_rmse_train,     \
    sixtrials_rmse_test, sixtrials_nrmse_train, sixtrials_nrmse_test,   \
    sixtrials_data_train, sixtrials_data_test, sixtrials_trainPredict,  \
    sixtrials_testPredict , sixtrials_loss

def custom_algorithm_individual(which_model, subject_id, trial_id, layer, node, increments, data, X_variables, Y_variable, epochs, cb_list, range_, layers_nodes, learning_rate, window_size, step):
    variables = [X_variables + Y_variable][0]
    data = data[variables]
    
    # save within same layer and same node but 3 different trials
    fivetrials_r2_train = np.array([])
    fivetrials_r2_test = np.array([])
    fivetrials_rmse_train = np.array([])
    fivetrials_rmse_test = np.array([])
    fivetrials_nrmse_train = np.array([])
    fivetrials_nrmse_test = np.array([])
    fivetrials_data_train = list()
    fivetrials_data_test = list()
    fivetrials_trainPredict = list()
    fivetrials_testPredict = list()
    fivetrials_loss = list()
    
   
    """3-fold cross validation, with 33% being test set"""
    for i in range(range_[0],range_[1]):
        print('--------------------------------------------------------------- trial:', str(i))
        "get variables for calculation and prepare data"
        #create array that shows what should be used for test and training
        dataset_train, dataset_test, data_train, data_test = ppf.create_train_test_tensor(data, window_size, step, X_variables, Y_variable, range_, increments, i)
        
       
        """
        for element in dataset_test.as_numpy_iterator():
            b = element
            break
        
        for sample in dataset_train.take( 1 ):
            print( sample[ 0 ].shape )
            print( sample[ 1 ].shape )
        """
        
        
        model = ppf.custom_model(window_size, len(X_variables), layers_nodes, learning_rate)
        history = model.fit(dataset_train,
                     epochs=epochs,
                     verbose=1,
                     shuffle=False,
                     callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                     #validation_split = 0.2,
                     #validation_data = (testX, testY)
                     )
        
        # make predictions
        trainPredict = model.predict(dataset_train)
        testPredict = model.predict(dataset_test)
        """
        plt.figure()
        plt.title('Predicted Power vs Actual Power normalized-test')
        predicted_power = plt.plot(testPredict[:,0], label = 'predicted_power', color = 'r')
        actual_power = plt.plot(data_test[window_size-1:].power_filtered.reset_index(drop = True), label = 'measured_power', color = 'b')
        plt.legend()
       """
       
        # evaluate the keras model
        r2_train, rmse_train, nrmse_train = ppf.cost(data_train[window_size-1:].power_filtered, trainPredict[:,0])
        r2_test, rmse_test, nrmse_test = ppf.cost(data_test[window_size-1:].power_filtered, testPredict[:,0])
        
      
        # store results of 3 different trials
        fivetrials_r2_train = np.append(fivetrials_r2_train, r2_train)
        fivetrials_r2_test = np.append(fivetrials_r2_test, r2_test)
        fivetrials_rmse_train = np.append(fivetrials_rmse_train, rmse_train)
        fivetrials_rmse_test = np.append(fivetrials_rmse_test, rmse_test)
        fivetrials_nrmse_train = np.append(fivetrials_nrmse_train, nrmse_train)
        fivetrials_nrmse_test = np.append(fivetrials_nrmse_test, nrmse_test)
        fivetrials_data_train.append(data_train)
        fivetrials_data_test.append(data_test)
        fivetrials_trainPredict.append(trainPredict)
        fivetrials_testPredict.append(testPredict)
        fivetrials_loss.append(history.history['loss'])
        #fivetrials_val_loss.append(history.history['val_loss'])
    
    return fivetrials_r2_train, fivetrials_r2_test, fivetrials_rmse_train,     \
        fivetrials_rmse_test, fivetrials_nrmse_train, fivetrials_nrmse_test,    \
        fivetrials_data_train, fivetrials_data_test,     \
        fivetrials_trainPredict, fivetrials_testPredict, fivetrials_loss
           

def kerastuneralgorithm_combined(data, X_variables, Y_variable, increments, tuner_epochs, dir_save, cb_list, which_model, subject_id, max_trials, save_flag, save_name, window_size, step, range_):

        sixtrials_name = list()
        
        for i in range(range_[0], range_[1]):
            print('--------------------------------------------------------------- trial:', str(i))
            "get variables for calculation and prepare data"
            #create array that shows what should be used for test and training
            dataset_train, dataset_test, data_train, data_test = ppf.create_train_test_tensor(data, window_size, step, X_variables, Y_variable, range_, increments, i)
        
       
            
            """fun starts here"""
            if which_model == "bayesiansearch":
                tuner = kerastuner.BayesianOptimization(
                                         ppf.tuner_model_shallow_combined,
                                         objective='val_loss',
                                         max_trials = max_trials,  
                                         directory = dir_save,
                                         project_name = 'bayesian_all' + str(i)
                                         )
                
            if which_model == "randomsearch":
                tuner = kerastuner.RandomSearch(
                    ppf.build_model_stack,
                    objective = 'val_loss',
                    max_trials = 5,
                    executions_per_trial = 3,
                    directory = dir_save,
                    project_name = 'randomsearch_all')
            
            if which_model == "hyperband":
                tuner = kerastuner.kt.BayesianOptimization(
                                         ppf.hyperband_model,
                                         objective='val_loss',
                                         #max_epochs=epochs,  
                                         factor = 3,
                                         directory = dir_save,
                                         project_name = 'hyperband_all' + str(i)
                                         )
            
            tuner.search_space_summary()
                         
            # train all the models
            tuner.search(dataset_train,
                        verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
                        epochs=tuner_epochs,
                        callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                        validation_data = dataset_test,
                        )
            
            # create name dynamically "subjectid_trial_id_numlayers_units1_units2_unitsN"
            best_hps = tuner.get_best_hyperparameters(1)[0]
            
            # this will be needed for deep neural network tests
            """
            num_layers = best_hps.get('num_layers') + 1
            
            onetrial_name = str(subject_id) + '_' + str(i) + '_' + str(num_layers)
            for t in range(0,num_layers-1):
                    onetrial_name = onetrial_name + '_' + str(best_hps.get('units_' + str(t)))
            onetrial_name = onetrial_name + '_' + str(best_hps.get('units_end')) + '_' + str(best_hps.get('learning_rate'))
            """
            
            num_layers = 1
            onetrial_name = str(subject_id) + '_' + str(i) + '_' + str(num_layers) + '_' + str(best_hps.get('units_0')) + '_' + str(best_hps.get('learning_rate'))
            
            sixtrials_name.append(onetrial_name)
              
            
            
            if save_flag:
                print('should be saving now')
                #save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Hyperband/CombinedSubject' + str(subject_id) + '_' + '.pkl'
                save_path = dir_save + save_name + '.pkl'
                print(save_path)
                
                with open(save_path, 'wb') as f:
                    pickle.dump([sixtrials_name], f)  
                print('should be saved now ')
          
        return sixtrials_name
           


    
        
def kerastuneralgorithm_individual(data, X_variables, Y_variable, increments, epochs, dir_save, cb_list, which_model,subject_id, max_trials, save_flag, save_name, window_size, step, range_):
    # save within same layer and same node but 6 different trials
    #create array that shows what should be used for test and training
    fivetrials_name = list()
    for i in range(range_[0], range_[1]):
        print('--------------------------------------------------------------- trial:', str(i))
        "get variables for calculation and prepare data"
        #create array that shows what should be used for test and training
        dataset_train, dataset_test, data_train, data_test = ppf.create_train_test_tensor(data, window_size, step, X_variables, Y_variable, range_, increments, i)
        
        """fun starts here"""
        if which_model == "bayesiansearch":
            tuner = kerastuner.BayesianOptimization(
                                     ppf.tuner_model_shallow_individual,
                                     objective='val_loss',
                                     max_trials=max_trials,  
                                     directory = dir_save,
                                     project_name = 'bayesian_all' + str(i)
                                     )
            
        if which_model == "randomsearch":
            tuner = kerastuner.RandomSearch(
                ppf.tuner_model,
                objective = 'val_loss',
                max_trials = 5,
                executions_per_trial = 3,
                directory = dir_save,
                project_name = 'randomsearch_all')
        
        if which_model == "hyperband":
            tuner = kerastuner.kt.BayesianOptimization(
                                     ppf.tuner_model,
                                     objective='val_loss',
                                     #max_epochs=epochs,  
                                     factor = 3,
                                     directory = dir_save,
                                     project_name = 'hyperband_all' + str(i)
                                     )
        
        tuner.search_space_summary()
                     
        # train all the models
        tuner.search(dataset_train,
                    verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
                    epochs=epochs,
                    callbacks=cb_list,  # if you have callbacks like tensorboard, they go here.
                    validation_data = dataset_test
                    )
        
        
        
        best_hps = tuner.get_best_hyperparameters(1)[0]
        
        """
        num_layers = best_hps.get('num_layers') + 1
        
        onetrial_name = str(subject_id) + '_' + str(i) + '_' + str(num_layers)
        for t in range(0,num_layers-1):
                onetrial_name = onetrial_name + '_' + str(best_hps.get('units_' + str(t)))
        onetrial_name = onetrial_name + '_' + str(best_hps.get('units_end')) + '_' + str(best_hps.get('learning_rate'))
        """
        
        num_layers = 1
        onetrial_name = str(subject_id) + '_' + str(i) + '_' + str(num_layers) + '_' + str(best_hps.get('units_0')) + '_' + str(best_hps.get('learning_rate'))
        
        fivetrials_name.append(onetrial_name)
        
        
            
            
        
        
        if save_flag:
            print('Hopped into saving mode')
           
            print('should be saving now')
            #save_path = dir_root + 'OpenLoop/ResultsPython/FindAlgorithm/Hyperband/CombinedSubject' + str(subject_id) + '_' + '.pkl'
            save_path = dir_save + save_name + '.pkl'
            print(save_path)
            
            with open(save_path, 'wb') as f:
                pickle.dump([fivetrials_name], f)  
            print('should be saved now')   
                
          
    return fivetrials_name       
        
   