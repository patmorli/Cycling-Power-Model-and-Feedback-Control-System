#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:10:20 2022

@author: patrickmayerhofer

visualize_model
"""

import PredictPower_functions as ppf
from keras_visualizer import visualizer 
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


layers_nodes = [8] # this arrray will be the number of nodes for each layer. more numbers -> more layers
learning_rate = 0.01
X_variables = ['rpm_filtered','rpmdot_filtered', 'gear_ratio'] #'rpmdot_filtered'
Y_variable = ['power_filtered']
window_size = 8

model = models.Sequential([  
    layers.Dense(8, activation='relu', input_shape=(2,)),   
    layers.Dense(1, activation='relu')])  


model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(8, 2,1)))
model.add(layers.Dense(1))
model.compile(optimizer = Adam(learning_rate = 0.01), loss='mse', metrics = ['mse'])
print(model.summary())


model = ppf.custom_model(window_size, len(X_variables), layers_nodes, learning_rate)

visualizer(model, format='png', view=True)
