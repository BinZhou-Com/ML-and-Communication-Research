# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:20:19 2019

@author: user
"""
#%%
'''
    'imulation v' theory
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

'''
    MAP Decoder v theory
'''


