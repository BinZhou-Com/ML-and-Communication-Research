# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:54:24 2019

@author: Eduardo D
"""
# BPSK digital modulation
#%%
# Import libraries
from numpy import *
from scipy.special import erfc # complementary error function
import matplotlib.pyplot as plt

#%% Initialize variables
# Model: communication system that sends N bits per seconde through a noisy channel
# the bits have constant energy, however the channel has variable power

# Transmission
N = 512 # size of the packet
r = 256 # bit rate (bits/s)

p = random.randint(0,2,size=N) # random packet of size N

f = r # hypothesis: sampling rate equals to bit rate
Tb = r/N # bit duration

T = N/r # sampling time duration (period)

t = arange(0,T,1/f) # time space

#%% Error analysis parameters
SNRdbmin = 0
SNRdbmax = 10 
Eb_No_dB = arange(SNRdbmin, SNRdbmax+1) # signal to noise ration (SNR) in (dB)
SNR = 10**(Eb_No_dB/10.0) # signal to noise ratio (linear)
Eb = Tb # energy per bit (J) - chosen as Tb to normalise Psignal
Psignal = Eb/Tb # constant and equals to 1 (linear)
PsignaldB = 10*log(Psignal)
No = Eb/SNR
A_BER = 0.5*erfc(sqrt(SNR)) # analytical BER

#%% Modulation - time domain
# BPSK : 0/1 -> -1/1
phi =  sqrt(2*Psignal)*cos(2*pi*f*t) # basis function
u = (2*(p>0.5)-1)*phi

#%% Fourrier Transform - frequency domain mean 0

#%% Frequency shifting to channel frequency

#%% Add Noise to the channel
# Channel characteristics
# channeL TYPE: 1 - AWGN; 2 - bernoulli ...
# Noise AWGN
CTYPE = 1



w = empty([size(var),N])
i = 0
for i in range(0,size(w,0)):
    w[i] = sqrt(var[i])*random.randn(N)
    
x = u + w # output signal + noise

#%% Receiver - Decoder

#%% Intermediate plots
# intermediate plots
plt.figure(1)
plt.plot(t/T,u,'b')
plt.plot(t/T,w[9],'g')
plt.plot(t/T,x[9],'r')
plt.grid(True)
plt.title('Modulated signal')
plt.xlabel('Normalized time')
plt.ylabel('Pulse amplitude')
plt.show()

#%% Decoder

# simplest decodification
y = sign(x)

# find erroneous symbols


#plt.semilogy(Eb_No_dB, Pe,'r',Eb_No_dB, BER,'s')
plt.semilogy(Eb_No_dB, A_BER,'r',linewidth=2)
plt.semilogy(Eb_No_dB, A_BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Sig. to noise ratio (Eb/No (dB))')
plt.ylabel('BER')
plt.show()