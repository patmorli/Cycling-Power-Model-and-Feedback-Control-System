#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:09:32 2021

@author: patrickmayerhofer
"""
import pickle

def save_model_tuner(which_model, dir_save, sixtrials_tuner_results):
        save_path = dir_save + 'summary_results.pkl'
        
        with open(save_path, 'wb') as f:
            pickle.dump([sixtrials_tuner_results], f)  

def save_model_forloop():
    print('not implemented yet')