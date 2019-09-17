# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:01:42 2019

@author: user
"""
MAPError = np.empty([globalReps, len(SNR)])
for i_global in range(globalReps):
    for i_snr in range(np.size(SNR)):
        snr = SNR[i_snr]
        
        '''
               Generate channel Input
        '''
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        
        ''' 
            Channel Encoding: BPSK
        '''
        xflat = np.reshape(x, [-1])
        xBPSK = fn.BPSK(xflat)
        yflat = fn.AWGN(xBPSK,snr)
        ychannel = yflat.reshape(N,n) # noisy codewords
        
        '''
            MAP Decoder (minimum distance decoding)
        '''
        # soft decision (euclidian distance)
        y = ychannel
        MAP = np.empty([N,k]) # decoded
        for i in range(N):
            minDistWord = np.argmin(fn.euclidianDistance(possibleCodewords, y[i]), 0) # find word of minimum distance
            MAP[i] = messages[minDistWord]
            
        '''
            Error Calculation
        '''
        MAPError[i_global][i_snr] = fn.codeErrorFunction(MAP, u)
        
'''
    MC error treatment
'''
avgMAPError = np.average(MAPError, 0)

'''
    Save Data
'''
filename = './Data/MAP/MAP.pickle'
with open(filename,  'wb') as f:
    pickle.dump(avgMAPError, f)
    
    
#%%
markerlist = ['^', '']
linelist = ['', '--']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             theoreticalErrorBPSK], 
            ['Soft MAP Decoder', 'Uncoded BPSK'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MAP-vs-theory.png', bbox_inches='tight', dpi=300)

