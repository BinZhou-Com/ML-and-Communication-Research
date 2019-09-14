# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:02:48 2019

@author: user
"""
#%% Initialization
globalReps = 1000
MAPpredictTime = TicToc('MAPPredict')
globalError = np.empty([globalReps, len(pOptions)])
globalErrorHamming = np.empty([globalReps, len(pOptions)])
globalErrorMAP = np.empty([globalReps, len(pOptions)])
title = 'MAP'
MAPpredictTime.tic()
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
            minDistWord = np.argmin(np.sum(possibleCodewords!=y[i], 1), 0) # find word of minimum distance
            MAP[i] = messages[minDistWord]
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_p] = fn.codeErrorFunction(y, x)
        globalErrorMAP[i_global][i_p] = fn.bitErrorFunction(MAP, u)

MAPpredictTime.toc()
print('Total MAP predict time: ', MAPpredictTime.elapsed)
#% plot
        
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)

fig = plt.figure(figsize=(8, 6), dpi=80)

plt.plot(pOptions,avgGlobalError, color='k', linewidth=lineWidth, linestyle='--')
plt.plot(pOptions,avgGlobalErrorMAP, color='k', linewidth=lineWidth)

plt.grid(True, which="both")
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP Algorithm'])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('images/'+title+'/'+timestr+'_MAP.png', bbox_inches='tight', dpi=300)

