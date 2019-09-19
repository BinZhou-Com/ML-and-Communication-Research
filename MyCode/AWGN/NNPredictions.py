# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:34:10 2019

@author: user
"""
#%%
'''
    Single Prediction
'''
#title = 'MLNN'
#directory = 'Models/'+title+'/'
#lw = '[128,64,32,8]'
#elevado = 14
#numEpochs=2**elevado
fileName = 'MLNN_'+lw+'_Mep_'+str(numEpochs)+'.h5'
'MLNN_[128,64,32,8]_Mep_262144_20190918-205311.h5'
Decoder = tf.keras.models.load_model(directory+fileName) # load model weights
print("Loaded models from disk")

fn.MLNNSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title)

markerlist = ['', '^']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgMLNNError], 
            ['Soft MAP Decoder', '$M_{ep} = 2^{'+str(elevado)+'}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MLNN-vs-Decoder-Mep-'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)

#%%
'''
    Multi Prediction
'''


