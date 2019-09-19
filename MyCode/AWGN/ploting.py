# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:20:19 2019

@author: user
"""
#%%
'''
    Simulation vs theory
'''
markerlist = ['x', '']
linelist = ['', '--']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgGlobalError, 
             theoreticalErrorBPSK], 
            ['Simulation', 'Theory'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/simu-vs-theory.png', bbox_inches='tight', dpi=300)


#%% Plot and save results
'''
    MAP Decoder vs theory
'''
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

#%%
'''
    Single decoder predict vs MAP
'''
markerlist = ['', '^']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgMLNNError], 
            ['Soft MAP Decoder', '$M_{ep} = 2^{12}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MLNN-vs-Decoder-Mep-'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)


