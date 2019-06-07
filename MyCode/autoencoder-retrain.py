# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:47:36 2019

@author: user
"""
#%%
numEpochs = 2**16
directory = 'Retrain\AutoencoderArray\\'
fileName = 'i_1_AutoencoderArray_Mep_65536_p_0.03.h5'
Autoencoder = tf.keras.models.load_model(directory+'OriginalModel\\'+fileName)
print("Loaded models from disk")
path = directory+'TrainedModel\\'+fileName

'''
    Overall Settings/ Compilationl
'''
lossFunc = 'mse'
Autoencoder.compile(loss=lossFunc ,
              optimizer='adam',
              )
'''
    Summaries and checkpoints (to do)
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

Encoder = Autoencoder.layers[0]
Decoder = Autoencoder.layers[2]
#% Prediction
globalErrorAutoencoder=fn.arrayAutoencoderPrediction(Encoder, Decoder, pOptions, globalReps, N, n, k)

#% Plotting
figPath = directory+'AutoencoderArray_64-128-256_Mep_131072_p_0.03.png'
numEpochs = 2**17
plotBERp(globalErrorAutoencoder, 'Autoencoder')



