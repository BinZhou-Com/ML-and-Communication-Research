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
N = 10 # number of messages sent

# Channel
CTYPE = "BSC"
p = 0.01 # cross probability of BSC
e = stats.bernoulli.rvs(p, size=n) # BSC noise
He = fn.Hb(p)# binary entropy function
C = 1- He # channel capacity
#%% 
'''
       Generate channel Input
'''
u = stats.bernoulli.rvs(0.5,size=[N,k]) # input matrix

x = empty([N,n]) # code words

for i in range(N):
    x[i] = dot(u[i],G)%2 # codeword row vector

#%% Pass through BSC
''' 
    Channel
'''

xflat = reshape(x, [-1])

yflat = fn.BSC(xflat,p)

y = yflat.reshape(N,n) # noisy codewords

#%%
'''
   Codeword Error without post treatment
'''
E1cw, E1b = fn.errorFunction(D, x, u, name)

#%%
'''
    Decoding
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

E2cw, E2b = fn.errorFunction(D, x, u, name)


#%% MAP DECODER
'''
    MAP Decoder
'''
compare = fn.possibleCodewords(name)
dist = 9999*ones(size(y,0))
MAP = y
syndromeErrors = sum(S, 1)
pos = -1;
for i in range(N):
    if(syndromeErrors[i]!=0): # there is at least a bit error
        for word in compare:
            d = fn.hammingDistance(word, y[i])
            if(d < dist[i]):
                dist[i] = d
                MAP[i] = word
            if(d == dist[i] and random.rand() < 0.5):
                dist[i] = d

EcwMAP, EbMAP = fn.errorFunction(MAP, x, u, name)

#%% Rudimentary error corrections
'''
    Show quickly if the algo is working
'''

print("# of binary errors without correction: \n", E1b)
print("# of binary errors with Syndrome correction: \n", E2b)
print("# of binary errors with MAP correction: \n", EbMAP)

