# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:58:01 2019

@author: user
"""
'''
       Load libraries
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erfc  # complementary error function

import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K
from keras.utils import plot_model

import sys
sys.path.append('C:\\Users\\user\\Desktop\\GitHub\\PIR\\MyCode')
import my_functions as fn

import time
import os
from ttictoc import TicToc

import pickle

#Plot setup
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams["font.family"] = "Times New Roman"
letterSize = 8
markerSize = 3
lineWidth = 0.75

mpl.rc('xtick', labelsize=letterSize)
mpl.rc('ytick', labelsize=letterSize)
mpl.rc('axes', labelsize=letterSize)
mpl.rc('legend', fontsize=letterSize)

# width as measured in inkscape
width = 3.487
height = width / 1.618


#import matplotlib
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()

'''
       Load constants
'''
N = 100 # number of messages sent

# Message characteristics
## Hamming coding
m = 4 # parity bits
n = 2**m-1 # total bits of one codeword
k = n-m # data bits
n = 16
k = 8
d = n-k+1
R = np.log(k)/n # rate
name = np.array([n, k]) # Hamming(n,k)

Gdef = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
G = Gdef

messages, possibleCodewords = fn.possibleCodewordsG(name, G)
possibleRealCodewords = fn.BPSK(possibleCodewords)
# Channel
CTYPE = "AWGN"
# Error analysis parameters
SNRdbmin = 0
SNRdbmax = 10
Eb_No_dB = np.linspace(SNRdbmin, SNRdbmax, 20) # signal to noise ration (SNR) in (dB)
SNR = 10**(Eb_No_dB/10.0) # signal to noise ratio (linear)
BER = 0.5*erfc(np.sqrt(SNR)) # analytical BER
C = 1/2*np.log(1+SNR)# channel capacity
sig = np.sqrt(1/SNR) # scaling factor
#BPSK
A = 1 # dmin = 2A

# Simulation
globalReps = 10000

