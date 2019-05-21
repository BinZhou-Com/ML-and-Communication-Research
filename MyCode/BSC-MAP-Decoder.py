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
pOptions = np.arange(0.005, 0.11, 0.005) # cross probability of BSC
He = fn.Hb(pOptions)# binary entropy function
C = 1-He # channel capacity (R<C)

# Simulation
globalReps = 1000
globalError = np.empty([globalReps, len(pOptions)])
globalErrorHamming = np.empty([globalReps, len(pOptions)])
globalErrorMAP = np.empty([globalReps, len(pOptions)])
messages, possibleCodewords = fn.possibleCodewordsG(name, G)
#%% 
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        
        '''
               Generate channel Input
        '''
        
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G,)
        
        ''' 
            Channel
        '''
        
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        
        '''
            Decoding
        '''
      
        '''
            MAP Decoder (minimum distance decoding)
        '''

        MAP = np.empty([N,k])
        for i in range(N):
            minDistWord = np.argmin(sum(possibleCodewords!=y[i], 1), 0)
            MAP[i] = messages[minDistWord]
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_p] = fn.codeErrorFunction(y, x, name)
        globalErrorMAP[i_global][i_p] = fn.bitErrorFunction(MAP, u, name)
      

#%% Neural Networ decoder
'''
    DNN Decoder
'''
'''
    Training and validation data
'''

train_Size = 100 # all possible tuples
u_train_labels = fn.generateU(train_Size,k)
x_train = fn.generteCodeWord(train_Size, n, u_train_labels, G)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)
'''
    Custom Layer
'''
# value of p: optimal training statistics for neural based channel decoders (paper)
from keras import backend as K

def tensorBSC(x,p):
    noise = K.variable(value = np.random.rand(len(x))<p, dtype=np.float32)    
    result = K.add(noise, x)%2
    return result

def func_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)
    
'''
    Sequential Model: most simple tf MLNN model
'''
MLNN = tf.keras.Sequential([ # Array to define layers
              # Noise Layer
              layers.Lambda(tensorBSC,arguments = ['x',x, 'p',p],input_shape=(n,), output_shape=func_output_shape),
                      
                      '''
                      lambda x: 
                  (x + (np.random.rand(n) < np.random.uniform(low=0, high=0.5, size=1)).astype(dtype=np.float32))%2, 
                      '''
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(128, activation='relu', input_shape=(n,)),
              # Add another: L2
              layers.Dense(64, activation='relu'),
              # Add another: L3
              layers.Dense(32, activation='relu'),
              # Add layer with k output units:
              layers.Dense(k, activation='sigmoid')
])
    
'''
    Overall Settings
'''
MLNN.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
    Summaries (to do)
'''
MLNN.summary()

''' 
    Training
'''

numEpochs = 10000 #65000
batchSize = train_Size
history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
          validation_data=(x_val, u_val_labels))

plt.plot(history.history['loss'])

'''
    evaluate the inference-model
''' 

evaluation = MLNN.evaluate(x_val, u_val_labels, batch_size=batchSize)
#%%
'''
    Prediction
'''
globalErrorMLNN = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G,)
        prediction = MLNN.predict(x, batch_size=batchSize)
        # round predictions
        rounded = np.round(prediction)

        globalErrorMLNN[i_global][i_p] = fn.bitErrorFunction(rounded, u, name)

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


#avgGlobalErrorMLNN = np.average(globalErrorMLNN,0)
#plt.scatter(pOptions,avgGlobalErrorMLNN, color='g')


plt.grid(True)
plt.title('Error Analysis')
plt.xlabel('p')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP', 'DNN Decoder'])
plt.show()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
fig.savefig('images/test'+timestr+ '.png', bbox_inches='tight')