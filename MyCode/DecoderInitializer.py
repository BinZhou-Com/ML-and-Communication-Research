# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:55:00 2019

@author: user
"""
#%%
numEpochs = 2**10  #2**16 approx 65000
batchSize = 256 # Mini batch size
train_p = 0.07
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'MLNN'
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'

