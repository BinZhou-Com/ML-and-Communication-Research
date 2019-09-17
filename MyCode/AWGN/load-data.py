# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:10:56 2019

@author: user
"""
'''
    Uncoded imulation
'''
filename = './Data/simu/simu-vs-theory.pickle'
with open(filename, 'rb') as f:
    avgGlobalError, theoreticalErrorBPSK = pickle.load(f)
    
'''
    MAP Avg Error
'''
filename = './Data/MAP/MAP.pickle'
with open(filename, 'rb') as f:
    avgMAPError = pickle.load(f)