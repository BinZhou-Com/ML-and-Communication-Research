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