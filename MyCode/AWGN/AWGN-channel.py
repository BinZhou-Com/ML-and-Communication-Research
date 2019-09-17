# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:02:24 2019

@author: user
"""

#%% Analystical BER vs SNR
# BPSK
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


#%% Plot and save results
#fig = plt.figure(figsize=(8, 6))

markerlist = ['x', '']
linelist = ['', '--']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgGlobalError, 
             theoreticalErrorBPSK], 
            ['Uncoded BPSK', 'Theory'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/simu-vs-theory.png', bbox_inches='tight', dpi=300)

#%%
'''
fig = plt.figure(figsize=(8, 6))
xlist = [Eb_No_dB,Eb_No_dB]
ylist =  [avgGlobalError, theoreticalErrorBPSK]
for i,array in enumerate(xlist):
    plt.plot(xlist[i],ylist[i], color=colorlist[i], linewidth=lineWidth,
    linestyle=linelist[i], marker=markerlist[i], markersize=markerSize)

legend = ['Simulation', 'Theory']
X="Eb/No (dB)"
Y="BER"
plt.grid(True, which="both")
plt.xlabel(X)
plt.ylabel(Y)
plt.yscale('log')
plt.legend(legend)
plt.xlim([SNRdbmin, SNRdbmax])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('test2.png', bbox_inches='tight', dpi=300)
'''