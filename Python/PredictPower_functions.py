#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:34:00 2021

@author: patrickmayerhofer
"""
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
from numpy import zeros, ones
import math
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean 

# keras tuner

'''
initial_learning_rate = 1e-1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=500,
    decay_rate=0.97,
    staircase=True)
'''

"""creates test and train tensor with windows and given X and Y variables"""
def create_train_test_tensor(data, window_size, step, X_variables, Y_variable, range_, increments, i):
    """creates windows of a given window size and step length. Data should be
    dataframe, and X_variables, and Y_variable decides which variables to use"""
    def create_windows(data, window_size, step, X_variables, Y_variable):
        
        data_windows_input = list()
        data_windows_label = list()
        for i in range(0, len(data)-window_size +1, step):
            v = data.iloc[i:(i + window_size)]
            data_windows_input.append(v[X_variables])
            data_windows_label.append(v[Y_variable].iloc[-1:])
         
        input_data = np.reshape(np.asarray(data_windows_input), (1, len(data_windows_input),window_size, len(X_variables)))
        output_data = np.reshape(np.asarray(data_windows_label),(1, len(data_windows_label)))
         
        return input_data, output_data
    
    ones_matrix_test = zeros(increments[range_[1]])
    ones_matrix_train = ones(increments[range_[1]])
    for u in range(increments[i],increments[i+1]-1):
        ones_matrix_test[u] = 1;
        ones_matrix_train[u] = 0;
    
    # divide in train and data set
    data_test = data[ones_matrix_test>0]
    data_train = data[ones_matrix_train>0]
    
    input_data, output_data = create_windows(data_train, window_size, step, X_variables, Y_variable)
    dataset_train = tf.data.Dataset.from_tensor_slices((input_data, output_data))             
    input_data, output_data = create_windows(data_test, window_size, step, X_variables, Y_variable)
    dataset_test = tf.data.Dataset.from_tensor_slices((input_data, output_data))             
        
    return  dataset_train, dataset_test, data_train, data_test  
        

def save_data(summary_file, subject_id, trial_id, r2_train, r2_test, rmse_train, rmse_test):
    # save_data, Patrick Mayerhofer, September 2021
    # stores the data to the summary file that will be saved later
    if trial_id == 1:
        summary_file.loc[subject_id-1, 'r2_train_python_nn_1'] = r2_train
        summary_file.loc[subject_id-1,'r2_test_python_nn_1'] = r2_test
        summary_file.loc[subject_id-1, 'rmse_train_python_nn_1'] = rmse_train
        summary_file.loc[subject_id-1,'rmse_test_python_nn_1'] = rmse_test       
    elif trial_id == 2:
        summary_file.loc[subject_id-1,'r2_train_python_nn_2'] = r2_train
        summary_file.loc[subject_id-1,'r2_test_python_nn_2'] = r2_test
        summary_file.loc[subject_id-1,'rmse_train_python_nn_2'] = rmse_train
        summary_file.loc[subject_id-1,'rmse_test_python_nn_2'] = rmse_test
    elif trial_id == 3:
        summary_file.loc[subject_id-1,'r2_train_python_nn_3'] = r2_train
        summary_file.loc[subject_id-1,'r2_test_python_nn_3'] = r2_test
        summary_file.loc[subject_id-1,'rmse_train_python_nn_3'] = rmse_train
        summary_file.loc[subject_id-1,'rmse_test_python_nn_3'] = rmse_test
    return summary_file


def plot_data(data):
    plt.figure()
    plt.plot(data.time, data.bpm)
    plt.plot(data.time, data.rpm)
    
    plt.figure()
    plt.plot(data.time, data.power)
    
    plt.figure()
    plt.plot(data.time, data.gpsSpeed) 
    
def get_X_Y(data, X_variables, Y_variable):
    # convert integer values to floating point values
    data = data.astype('float32')
    
    # create trainX, trainY, textX, and textY
    X = data[X_variables]
    Y = data[Y_variable]
    
    # convert to numpy
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    
    # reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    return X, Y

def load_and_organize_data_combined(subject_id, trial_id, dir_root, normalize_flag, summary_file, fbracket):
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
    
    
    """create indizes for 3 same-sized test sets from each trial"""
    # create 3 test sets per trial
    l1_ = len(data1)
    l1_3 = round(l1_/3)
    increments1 = (0, l1_3,l1_3*2, l1_-1)
    
    l2_ = len(data2)
    l2_3 = round(l2_/3)
    increments2 = (0, l2_3,l2_3*2, l2_-1)
    
    # merge data1 and data2 and increments
    data = [data1, data2]
    data = pd.concat(data, ignore_index = True)
    increments2 = tuple([i + l1_ for i in increments2])
    increments = increments1 + increments2[1:len(increments2)]
    increments = list(increments)
    for i in range(len(increments)-1):
        increments[i+1] = increments[i+1] +1
        
        
        
    return increments, data

def load_and_organize_data_individual(subject_id, trial_id, dir_root, normalize_flag):
    "load csv files"
    filename = 'Subject' + str(subject_id) + '_' + str(trial_id)
    dir_load_file = dir_root + 'OpenLoop/CleanedCSV/' + filename + '.csv'
    
    # load data
    data = pd.read_csv(dir_load_file)
    
    if normalize_flag:
           data.power_filtered = (data.power_filtered-data.power_filtered.min())/(data.power_filtered.max()-data.power_filtered.min())                       
    l = len(data)
    l3 = round(l/3)
    increments = (0, l3, l3*2, l)
    
    return increments, data

def get_X_Y(data, X_variables, Y_variable):
        # convert integer values to floating point values
        data = data.astype('float32')
        
        # create trainX, trainY, textX, and textY
        X = data[X_variables]
        Y = data[Y_variable]
        
        # convert to numpy
        X = X.to_numpy()
        Y = Y.to_numpy()
        
        
        # reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        
        return X, Y

def prepare_dataset(increments, i, data, X_variables, Y_variable):
    def get_X_Y(data, X_variables, Y_variable):
        # convert integer values to floating point values
        data = data.astype('float32')
        
        # create trainX, trainY, textX, and textY
        X = data[X_variables]
        Y = data[Y_variable]
        
        # convert to numpy
        X = X.to_numpy()
        Y = Y.to_numpy()
        
        
        # reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        
        return X, Y
    
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
    trainX, trainY = get_X_Y(data_train, X_variables, Y_variable)
    testX, testY = get_X_Y(data_test, X_variables, Y_variable)
    
    return trainX, trainY, testX, testY, data_train, data_test

def cost(x, f):
    # x is the measured data set
    # f is the modelled data set
    x = x.reset_index(drop=True)
    r2 = r2_score(x, f)
    r2_alternative = 1 - sum((x-f)**2)/sum(x**2)
    mse = mean_squared_error(x, f)
    rmse = math.sqrt(mse)
    nrmse = rmse/mean(x)
    
    return r2_alternative, rmse, nrmse

def r2_alternative(x, f):
    # x is the measured data set
    # f is the modelled data set
    r2_alternative = 1 - sum((x-f[:,0])**2)/sum(x**2)
    
    return r2_alternative

def norm_mean_error(x, f):
    # x is the measured data set
    # f is the modelled data set
    mean_err = sum(x-f[:,0])/len(f)
    norm_mean_err = mean_err/mean(x)
    return norm_mean_err

def plot_squared_error_over_time(x,f):
    squared_error = list()
    x = x.reset_index(drop=True)
    for i in range(0,len(x)):
        squared_error.append(float(x[i]-f[i])**2) 
    return squared_error

def get_previous(data, column_name):
    current = data[column_name]
    previous = pd.Series(current[0])
    previous = previous.append(current)
    previous.drop(previous.tail(1).index, inplace = True)
    previous =  previous.reset_index(drop = True)
    
    previous.name = column_name + '_prev'
    data[previous.name] = previous
    return data
    
def get_dot(data, column_name):
    return 0

def create_model(trainX):
    model = Sequential()
    model.add(LSTM(32,activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))
    #model.add(LSTM(8,activation='relu'))

    model.add(LSTM(32,activation='tanh'))
    model.add(LSTM(32,activation='tanh'))
    model.add(LSTM(32,activation='tanh'))
    model.add(LSTM(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['MeanSquaredError'])
    return model

def create_noRNN_model(trainX):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mse', metrics = ['MeanSquaredError'])
    return model


def custom_model(n_timestamps, n_variables, layers_nodes, learning_rate):
    
    # model_specifics is a list where the size represents the number of
    # layers and the numbers the nodes in each layer
    model = Sequential()
    # if only one layer, don't return full sequence, otherwise do
    if len(layers_nodes) == 1:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=False, input_shape=(n_timestamps, n_variables)))
    else:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True, input_shape=(n_timestamps, n_variables)))
   
        if len(layers_nodes) > 2:
            for i in range(0,len(layers_nodes)-2):
                model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences = True))
        model.add(LSTM(layers_nodes[-1:][0], activation = 'relu', return_sequences = False))
        
    model.add(layers.Dense(1))
    
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss='mse', metrics = ['mse'])
    print(model.summary())
    
    """
    model = Sequential()
    model.add(LSTM(16, activation='relu', return_sequences=False, input_shape=(n_timestamps, n_variables)))
    #model.add(LSTM(16, activation='relu', return_sequences=False))
    
    #model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss='mse', metrics = ['mse'])
    print(model.summary())
    """
    return model



def tuner_model_shallow_combined(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_0',min_value=16,
                                     max_value=1024,
                                     step=16), 
                   return_sequences = True,
         activation='relu', input_shape=(8,3))) 
    
    model.add(layers.Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
         hp.Choice('learning_rate',
                   values=[1e-1, 1e-2, 1e-3])))
    model.summary()
    
    return model

def tuner_model_deep_combined(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_0',min_value=64,
                                     max_value=256,
                                     step=64), 
                   return_sequences = True,
         activation='relu', input_shape=(8,3)))
    
    for i in range(hp.Int('num_layers', 0,2,step=1)):
         model.add(LSTM(hp.Int('units_' + str(i),
                                             min_value=64,
                                             max_value=256,
                                             step=64),
                        activation='relu', return_sequences=True))
   
    model.add(LSTM(hp.Int('units_end', 
                          min_value=64,
                          max_value=256,
                          step=64),
                          activation='relu', return_sequences=False))    
    
    model.add(layers.Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
         hp.Choice('learning_rate',
                   values=[1e-1, 1e-2, 1e-3])))
    model.summary()
    
    return model

def tuner_model_shallow_individual(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_0',min_value=16,
                                     max_value=1024,
                                     step=16), 
                   return_sequences = True,
         activation='relu', input_shape=(8,1))) 
    
    model.add(layers.Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
         hp.Choice('learning_rate',
                   values=[1e-1, 1e-2, 1e-3])))
    model.summary()
    
    return model

def tuner_model_deep_individual(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_0',min_value=16,
                                     max_value=64,
                                     step=16), 
                   return_sequences = True,
         activation='relu', input_shape=(8,2)))
    
    for i in range(hp.Int('num_layers', 10, 16)):
         model.add(LSTM(hp.Int('units_' + str(i),
                                             min_value=16,
                                             max_value=64,
                                             step=16),
                        activation='relu', return_sequences=True))
    
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
         hp.Choice('learning_rate',
                   values=[1e-1, 1e-2, 1e-3])))
    model.summary()
    
    return model
    

def build_model_sentdex_tuner_cycling2(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    
    model.add(Dense(hp.Int('input_units',2,32,2), input_shape=(1, 3)))
    model.add(Activation('linear'))
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=2,
                                            max_value=32,
                                            step=2)))
        model.add(Activation('linear'))   

    model.add(layers.Dense(1))
    model.add(Activation('linear'))
    
    model.summary()

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["MeanSquaredError"])
    
    return model

def build_model_sentdex_tuner_cycling3(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    
    model.add(LSTM(hp.Int('input_units',2,32,2), input_shape=(1, 3), return_sequences=True))
    model.add(Activation('tanh'))
    
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(LSTM(hp.Int('units_' + str(i),
                                            min_value=2,
                                            max_value=32,
                                            step=2),
                                            return_sequences=True))
        model.add(Activation('tanh'))   
    
    model.add(LSTM(hp.Int('units_' + str(i),
                                            min_value=2,
                                            max_value=32,
                                            step=2)))
    
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=2,
                                            max_value=32,
                                            step=2)))
        model.add(Activation('tanh')) 
        
    model.add(layers.Dense(1))
    model.add(Activation('linear'))
    
    model.summary()

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["MeanSquaredError"])
    
    return model

def build_model_dynamic(trainX, best_model):
    model = Sequential()
    model.add(Dense(best_model['input_units'],input_shape=(trainX.shape[1], trainX.shape[2]), activation = 'relu'))
    # for the number of layers
    for i in range(best_model['num_layers']+1):
        model.add(Dense(best_model['units_' + str(i)], activation = 'relu'))
    
    model.add(layers.Dense(1, activation = 'linear'))
    
    model.summary()

    model.compile(optimizer="adam",
                  loss="mse",
                  #metrics=["MeanSquaredError"]
                  )   
    return model


def build_model_stack(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=8,
                                     max_value=64,
                                     step=8), 
                   return_sequences = True,
         activation='relu', input_shape=(1,3)))
    
    for i in range(hp.Int('num_layers', 1, 5)):
         model.add(LSTM(hp.Int('units_' + str(i),
                                             min_value=8,
                                             max_value=64,
                                             step=8),
                        activation='relu', return_sequences=True))
    
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
         hp.Choice('learning_rate',
                   values=[1e-2, 1e-3, 1e-4])))
    model.summary()
    
    return model

def build_model_stack_original(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), 
               activation='relu', input_shape=(1,3)))
    model.add(Dense(units=hp.Int('units',min_value=32,
                                    max_value=512,
                                    step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])))
    return model
        
    