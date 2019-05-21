# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:02:48 2019

@author: user
"""

#%% 
'''
       Load libraries
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import erfc  # complementary error function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

import sys
sys.path.append('C:\\Users\\user\\Desktop\\GitHub\\PIR\\MyCode')
import my_functions as fn

#%%
''' 
    Classes
'''
class MyError:
    def __init__(self, vec):
        self.out = vec

#%% 
'''
       Load constants
'''

# Message characteristics
## Hamming coding
m = 4 # parity bits
n = 2**m-1 # total bits of one codeword
k = n-m # data bits

n = 16
k = 8

d = n-k+1
R = k/n # rate
name = np.array([n, k]) # Hamming(n,k)
H = fn.parityCheckMatrix(name) # systematic form
#G = fn.matrixGenerator(H,name)

Gdef = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
G = Gdef

#print("Parity check: \n",dot(H.T,G)%2)
dmin = 3

# input
N = 100 # number of messages sent

# Channel
CTYPE = "BSC"
pOptions = np.arange(0.01, 0.51, 0.01) # cross probability of BSC
He = fn.Hb(pOptions)# binary entropy function
C = 1-He # channel capacity (R<C)

# Simulation
globalReps = 100
globalError = np.empty([globalReps, len(pOptions)])
globalErrorHamming = np.empty([globalReps, len(pOptions)])
globalErrorMAP = np.empty([globalReps, len(pOptions)])
possibleCodewords = fn.possibleCodewordsG(name, G)
#%% 
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        
        '''
               Generate channel Input
        '''
        
        u = stats.bernoulli.rvs(0.5,size=[N,k]) # input message matrix
        x = np.empty([N,n]) # code words
        for i in range(N):
            x[i] = np.dot(u[i],G)%2 # codeword row vector
        
        ''' 
            Channel
        '''
        
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
       
        '''
            MAP Decoder (minimum distance decoding)
        '''

        MAP = y.copy()
        for i in range(N):
            minDistWord = np.argmin(sum(possibleCodewords!=y[i], 1), 0)
            MAP[i] = possibleCodewords[minDistWord]
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_p] = fn.errorFunction(y, x, u, name)
        globalErrorMAP[i_global][i_p] = fn.errorFunction(MAP, x, u, name)
      
#%% plot
        
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)
fig = plt.figure(figsize=(8, 6), dpi=80)
'''
for i in range(np.size(globalError,0)):
    plt.scatter(pOptions,globalError[i], color='b')
    plt.scatter(pOptions,globalErrorMAP[i], color='r')
'''
plt.plot(pOptions,avgGlobalError, color='b')
plt.plot(pOptions,avgGlobalErrorMAP, color='r')
plt.grid(True)
plt.title('Error Analysis')
plt.xlabel('p')
plt.ylabel('BER')
plt.legend(['No Decoding', 'MAP'])
plt.show()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
fig.savefig('images/test'+timestr+ '.png', bbox_inches='tight')

#%% Neural Networ decoder
'''
    DNN Decoder
'''
'''
       Sequential Model: most simple tf MLNN model
'''
MLNN = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(n, activation='relu', input_shape=(n,)),
              # Add another: L2
              layers.Dense(n, activation='relu'),
              # Add another: L3
              layers.Dense(n, activation='relu'),
              # Add a softmax layer with k output units:
              layers.Dense(k, activation='sigmoid')
])
    
''' 
       Training model
'''

MLNN.compile(optimizer=tf.train.AdamOptimizer(0.001),
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])

'''
    Training data
'''
data = x.copy()
labels = u.copy(())



