# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:02:48 2019

@author: user
"""

#%% 
'''
       Load libraries
'''

from numpy import *
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import erfc  # complementary error function
import pyldpc

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

import sys
sys.path.append('C:\\Users\\user\\Desktop\\GitHub\\PIR\\MyCode')
import my_functions as fn


#%% 
'''
       Load constants
'''

# Message characteristics
## Hamming coding
m = 4 # parity bits
n = 2**m-1 # total bits of one codeword
k = n-m # data bits
d = n-k+1
R = k/n # rate
name = array([n, k]) # Hamming(n,k)
H = fn.parityCheckMatrix(name) # systematic form
G = fn.matrixGenerator(H,name)
print("Parity check: \n",dot(G,H.T)%2)
dmin = 3

# input
N = 100 # number of messages sent

# Channel
CTYPE = "BSC"
pOptions = arange(0.05, 0.55, 0.05) # cross probability of BSC
He = fn.Hb(pOptions)# binary entropy function
C = 1-He # channel capacity (R<C)

# Simulation
globalReps = 5
globalError = empty([globalReps, len(pOptions)])
globalErrorHamming = empty([globalReps, len(pOptions)])
globalErrorMAP = empty([globalReps, len(pOptions)])
#%% 
counter2 = 0
counter1 = 0
for counter1 in range(globalReps):
    counter2 = 0
    for p in pOptions:
        '''
               Generate channel Input
        '''
        u = stats.bernoulli.rvs(0.5,size=[N,k]) # input message matrix
        
        x = empty([N,n]) # code words
        
        for i in range(N):
            x[i] = dot(u[i],G)%2 # codeword row vector
        
        
        ''' 
            Channel
        '''
        
        xflat = reshape(x, [-1])
        
        yflat = fn.BSC(xflat,p)
        
        y = yflat.reshape(N,n) # noisy codewords
        
        
        '''
           Codeword Error without post treatment
        '''
        E1cw, globalError[counter1][counter2] = fn.errorFunction(y, x, u, name)
        
        '''
        '''
       #     Decoding
        '''
        # Syndrome decoding
        S = fn.syndrome(y, H, name)
        
        D = y # error corrected codewords
        e = zeros([N,n])# bit error location
        # single error correction
        for i in range(N):
            if(sum(S[i,:])!=0):
                #print(i)
                index = sum(S[i,:] == H.T,1).argmax() # Find position where H matrix says there is an error
                e[i,int(index)] += 1
        e = e%2
        D = (D+e)%2 # decoded codewords
        
        E2cw, globalErrorHamming[counter1][counter2] = fn.errorFunction(D, x, u, name)
        
        '''
        '''
            MAP Decoder (minimum distance decoding)
        '''
        S = fn.syndrome(y, H, name)
        compare = fn.possibleCodewords(name)
        dist = 9999*ones(size(y,0))
        MAP = y
        syndromeErrors = sum(S, 1)
        for i in range(N):
            if(syndromeErrors[i]!=0): # there is at least a bit error
                for word in compare:
                    d = fn.hammingDistance(word, MAP[i])
                    if(d < dist[i]):
                        dist[i] = d
                        MAP[i] = word
                    if(d == dist[i] and random.rand() < 0.5):
                        dist[i] = d
                        MAP[i] = word
        
        EcwMAP, globalErrorMAP[counter1][counter2] = fn.errorFunction(MAP, x, u, name)
        counter2 += 1

#%%
'''
    Show quickly if the algo is working
'''

print("# of binary errors without correction: \n", globalError[counter1][counter2])
print("# of binary errors with Syndrome correction: \n", globalErrorHamming[counter1][counter2])
print("# of binary errors with MAP correction: \n", globalErrorMAP[counter1][counter2])
      
#%% plot

fig = plt.figure(figsize=(8, 6), dpi=80)
for i in range(size(globalError,0)):
    plt.scatter(pOptions,globalError[i], color='b')
    plt.scatter(pOptions,globalErrorMAP[i], color='r')

plt.grid(True)
plt.title('Error Analysis')
plt.xlabel('p')
plt.ylabel('BER')
plt.legend(['No Decoding', 'MAP'])
plt.show()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
fig.savefig('test'+timestr+ '.png', bbox_inches='tight')


