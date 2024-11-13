#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 08:32:21 2021

@author: patrickmayerhofer

AnandModelFunctions
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from tensorflow.keras import initializers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,LSTM,Activation
from keras.utils import np_utils
import csv
import os

initializer = initializers.GlorotNormal()


def build_model_CNN_2D(input_shape=(175,3,1),num_classes=5):
    model = Sequential()
    # first conv layer (zero padding required, but not mentioned in paper)
    model.add(Conv2D(100, kernel_size=(7,3), strides=2, kernel_initializer = initializer, padding = 'same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,1), strides= 2, padding = 'same'))
    # second conv layer
    model.add(Conv2D(200, kernel_size=(3,100), strides = 2, kernel_initializer = initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # third conv layer
    model.add(Conv2D(300, kernel_size=(3,200), strides = 1, kernel_initializer = initializer))
    model.add(Activation('relu'))
    # fourth conv layer 
    model.add(Conv2D(300, kernel_size=(3,300), strides = 1, kernel_initializer = initializer))
    model.add(Activation('relu'))
    # fifth conv layer 
    model.add(Conv2D(200, kernel_size=(3,300), strides = 1, kernel_initializer = initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    # Flatten layer, should have length 1600
    model.add(Flatten())
    
    # FC - 150 
    model.add(Dense(150))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    # output layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.summary()
	model.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])
	return model