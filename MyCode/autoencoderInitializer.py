# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:42:07 2019

@author: user
"""
#%%
'''
    DNN One Hot Model Autoencoder
'''
'''
    One hot training and validation data
'''
u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = messages.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)
#%%
'''
    Array training and validation data
'''
u_train_labels = messages.copy()
x_train_data = u_train_labels

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

#%%
'''
    Constants
'''
numEpochs = 2**13  #2**16 approx 65000
batchSize = 256
train_p = 0.0
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'Autoencoder1H'
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'

