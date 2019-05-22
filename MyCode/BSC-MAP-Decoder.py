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

import time

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
globalReps = 10
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
            minDistWord = np.argmin(np.sum(possibleCodewords!=y[i], 1), 0)
            MAP[i] = messages[minDistWord]
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_p] = fn.codeErrorFunction(y, x)
        globalErrorMAP[i_global][i_p] = fn.bitErrorFunction(MAP, u)

#%% Neural Networ decoder
'''
    DNN Decoder
'''
'''
    Training and validation data
'''
p = 0.07
from sklearn.utils import shuffle
train_Size = np.size(messages,0) # all possible tuples
u_train_labels, x_train = shuffle(messages.copy(), possibleCodewords.copy())

#TEST
x_train_flat = np.reshape(x_train, [-1])
y_train_flat = fn.BSC(x_train_flat,p)
y_train = y_train_flat.reshape(256,n) # noisy codewords

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)
'''
    Custom Layer
'''
from keras import backend as K

def tensorBSC(x):
    # value of p: optimal training statistics for neural based channel decoders (paper)
    p = K.constant(0.07,dtype=tf.float32)
    var = K.random_uniform(shape=(func_output_shape(x),), minval = 0.0, maxval=1.0)
    noise = K.less(var, p)
    noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noiseFloat, x)%2
    return result

def func_output_shape(x):
    shape = x.get_shape().as_list()[1]
    return shape
    
'''
    Sequential Model: most simple tf MLNN model
'''
MLNN = tf.keras.Sequential([ # Array to define layers
              # Noise Layer
              layers.Lambda(tensorBSC,input_shape=(n,), output_shape=(n,)),
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

MLNN.compile(loss='binary_crossentropy' ,
              optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # change accuracy to a BER function
              )
'''
    Summaries (to do)
'''
summary = MLNN.summary()

''' 
    Training
'''

numEpochs = 2**12  #2**16 aprprox 65000
batchSize = train_Size
history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize)
#history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
#          validation_data=(x_val, u_val_labels))

# summarize history for loss
trainingFig = plt.figure()
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
trainingFig.savefig('training_history/train'+timestr+ '.png', bbox_inches='tight')
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
        x = fn.generteCodeWord(N, n, u, G)
        
        prediction = MLNN.predict(x, batch_size=batchSize)
        # round predictions
        rounded = np.round(prediction)

        globalErrorMLNN[i_global][i_p] = fn.bitErrorFunction(rounded, u)

#%% plot
        
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)

fig = plt.figure(figsize=(8, 6), dpi=80)

plt.plot(pOptions,avgGlobalError, color='b')
plt.plot(pOptions,avgGlobalErrorMAP, color='r')

avgGlobalErrorMLNN = np.average(globalErrorMLNN,0)
plt.scatter(pOptions,avgGlobalErrorMLNN, color='g')

plt.grid(True)
plt.title('Error Analysis')
plt.xlabel('p')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP', 'DNN Decoder'])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.savefig('images/test'+timestr+ '.png', bbox_inches='tight')