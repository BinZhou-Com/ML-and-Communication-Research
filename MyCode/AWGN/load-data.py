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
    
'''
    MLNN Decoder
'''
filename = './Data/MLNN/MLNN_[128,64,32,8]_Mep_4096_.pickle'
with open(filename, 'rb') as f:
    avgMLNNError = pickle.load(f)