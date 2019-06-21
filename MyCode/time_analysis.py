# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:09:37 2019

@author: user
"""
#%%
'''
    MAP Decoder
'''
for i in range(10):
    exec(open("BSC-MAP-Decoder.py").read())

#%%
'''
    Array Decoder
'''
# NN characteristics:
predictTime = TicToc('Predict')
for i in range(10):
    predictTime.tic()
    globalErrorMLNN = fn.arrayDecoderPrediction(G, MLNNDecoder, pOptions, globalReps, N, n, k)
    predictTime.toc()
    print('Total predict time: ', predictTime.elapsed)
    plotBERp(globalErrorMLNN, 'Array Decoder')
    
    #%%
'''
    One-hot Decoder
'''
# NN characteristics:
predictTime = TicToc('Predict')
for i in range(10):
    predictTime.tic()
    globalErrorMLNN1H = fn.onehotDecoderPrediction(G, MLNN1H, pOptions, globalReps, N, n, k, messages)
    predictTime.toc()
    print('Total predict time: ', predictTime.elapsed)
    plotBERp(globalErrorMLNN1H, 'One-hot Decoder')
    
#%%
'''
    Array autoencoder
'''
# Load model
title = 'AutoencoderArray'
directory = 'Saved_Models/'+ title + '/'
fileName = directory + '20190617-102844_MAP_AutoencoderArray_Mep_65536_ptrain_003_logcosh.h5'
loadedModel = tf.keras.models.load_model(fileName)
print("Loaded model from disk")

Encoder = loadedModel.layers[0]
Decoder = loadedModel.layers[2]

predictTime = TicToc('Predict')
for i in range(10):
    predictTime.tic()
    globalErrorAA = fn.arrayAutoencoderPrediction(Encoder, Decoder, pOptions, globalReps, N, n, k)
    predictTime.toc()
    print('Total predict time: ', predictTime.elapsed)
    plotBERp(globalErrorAA, 'Array Autoencoder')

#%%
'''
    One-hot autoencoder
'''
# Load model
title = 'Autoencoder1H'
directory = 'Saved_Models/'+ title + '/'
fileName = directory + '20190621-110223_Autoencoder1H_Mep_65536_bs_256.h5'
loadedModel = tf.keras.models.load_model(fileName)
print("Loaded model from disk")

Encoder = loadedModel.layers[0]
Decoder = loadedModel.layers[2]
globalReps = 1000
predictTime = TicToc('Predict')
for i in range(10):
    predictTime.tic()
    globalErrorA1H = fn.onehotAutoencoderPrediction(Encoder, Decoder, messages, pOptions, globalReps, N, n, k)
    predictTime.toc()
    print('Total predict time: ', predictTime.elapsed)
    plotBERp(globalErrorA1H, 'One-hot Autoencoder')
