# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:28:03 2019

@author: Eduardo D
"""

from numpy import *
import matplotlib.pyplot as plt
from scipy.special import erfc # complementary error function

def plotit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t,u,'b')
    
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()

def scatterit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t,u, 'o')
    
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()

def sineWave(t, mean, amplitude, f):
    omega = 2 * pi * f # fs > 2f
    return amplitude * sin(omega * t + 0) + mean

def analog2binary(t, bCodification, inputSignal):
    # Example: we divide the amplitude in a scale of 8 bits and each sequence of 8 bits represents a point
    # in the end we should have t/T * 8 bits
    u = inputSignal
    M = len(u)
    x = 2**bCodification * (
    u - mean(u) - min(u - mean(u))) / max(u - mean(u) - min(u - mean(u)))
    
    x = x.astype(int)
    
    scatterit(t, x, "Quantized signal")
    b = ''  # strings of all bits togheter
    for i in range(M):
        str = "{0:b}".format(x[i])  # variable length source code word
        str = str.zfill(bCodification)  
        b = b + str  # transform b in an array of symbols 0's and 1's
    
    return b
    
def PSK2 (b, V):
    # 2 - PSK: 2 different phase representations. Maps binary symbols unsing the map 0 -> -1 and 1-> 1. Constellation of size 2
    um = empty(len(b))  # u - modulated (symbols modulated)
    for i in range(len(b)): # step function
        if (b[i] == '0'):
            um[i] = -1*V
        else:
            um[i] = 1*V
            
    return um

def pulseSignal(A, d): 
    # d is the discretization of the pulse
    pulse1 = A*ones(int(d/2))
    pulse2 = zeros(int(d/2))
    
    return concatenate((pulse1, pulse2), axis = None)
    
def constantSignal(A, d): 
    # d is the discretization of the pulse
    pulse1 = A*ones(int(d))
    return pulse1
    
    
def pulses2waveform (N,d,p,uk):
    uwv = empty(N*d)
    for i in range(len(uwv)):
        uwv[i] = uk[int(i/d)]*p[i%d]
    return uwv
    
def fourriertransform(x, interval, fc, BB):
    freq = fft.fftfreq(len(x), interval) # Frequency space
    xhat = fft.fft(x)
    fig, ax = plt.subplots()

    ax.plot(freq, xhat.real)
    ax.set_title('Waveform Fourrier Tranform')
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-fc-BB, fc+BB)
    
    return concatenate((freq, xhat), axis=0)
    
    
def Q(x):
    return 0.5*erfc(x/sqrt(2))

def MPSK_BER(M,Eb,No):
    n = log2(8)
    Es = n*Eb # energy per symbol: nEb where 2^n = M
    gamma_s = Es/No
    return 2*Q(sqrt(2*gamma_s)*sin(pi/M))/n
    
    