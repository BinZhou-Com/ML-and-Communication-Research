# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:02:24 2019

@author: user
"""

#%% Analystical BER vs SNR
# BPSK
globalReps = 10000
globalError = np.empty([globalReps, len(SNR)])
theoreticalErrorBPSK = 0.5*erfc(np.sqrt(SNR))
for i_global in range(globalReps):
    for i_snr in range(np.size(SNR)):
        snr = SNR[i_snr]
        
        '''
               Generate channel Input
        '''
        u = fn.generateU(N,k)
        x = u #fn.generteCodeWord(N, n, u, G)
        
        ''' 
            Channel Encoding: BPSK
        '''
        xflat = np.reshape(x, [-1])
        xBPSK = fn.BPSK(xflat)
        yflat = fn.AWGN(xBPSK,snr)
        ychannel = yflat.reshape(N,k) # noisy codewords
        
        '''
            Decoding
        '''
        # Hard deciion
        y = fn.decodeAWGN(ychannel)
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_snr] = fn.codeErrorFunction(y, x)
        
'''
    MC error treatment
'''
avgGlobalError = np.average(globalError, 0)

'''
    Save Data
'''
filename = './Data/simu/simu-vs-theory.pickle'
with open(filename,  'wb') as f:
    pickle.dump([avgGlobalError, theoreticalErrorBPSK], f)
