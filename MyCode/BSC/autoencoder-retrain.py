# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:47:36 2019

@author: user
"""
#%%

numEpochs = 2**10
directory = 'Retrain/Autoencoder1H/'
fileName = 'Autoencoder1H_Mep_131072_bs_256_-196_128_96_64_32-128.h5'
Autoencoder = tf.keras.models.load_model(directory+'OriginalModel\\'+fileName)
print("Loaded models from disk")
path = directory+'TrainedModel\\'+fileName
#%%
'''
    Overall Settings/ Compilationl
'''
lossFunc = 'logcosh'
Autoencoder.compile(loss=lossFunc ,
              optimizer='adam',
              metrics=[fn.metricBER1H]
              )
'''
    Summaries and checkpoints
'''
summary = Autoencoder.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpointPath, monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=checkpointPeriod)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = Autoencoder.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)

plotTraining(history)

'''
    Saving model
'''
Autoencoder.save(path)  # creates a HDF5 file
#%%
Encoder = Autoencoder.layers[0]
Decoder = Autoencoder.layers[2]
#% Prediction
#globalErrorAutoencoder=fn.arrayAutoencoderPrediction(Encoder, Decoder, pOptions, globalReps, N, n, k)
globalErrorAutoencoder=fn.onehotAutoencoderPrediction(Encoder, Decoder, messages, pOptions, globalReps, N, n, k)
#% Plotting
numEpochs = 2**17 + 2**10
figPath = directory+'images/Autoencoder1H_Mep_'+str(numEpochs)+'_bs_2048-128_p_train_0.png'
plotBERp(globalErrorAutoencoder, 'One-hot Autoencoder')



