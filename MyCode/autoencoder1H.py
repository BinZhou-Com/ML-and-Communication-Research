# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:56:03 2019

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
    Constants
'''
numEpochs = 2**14  #2**16 approx 65000
batchSize = trainSize 
train_p = 0.07
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'Autoencoder1H'

'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(128, activation='relu', input_shape=(k,), name='Input'),
        # Hidden Layer
        layers.Dense(64, activation='relu', name='EHL1'),
        # Hidden Layer
        layers.Dense(32, activation='relu', name='EHL2'),
        # Coded Layer
        layers.Dense(2*k, activation='sigmoid', name='Codedfloat'),
        # Rounded codeword
        layers.Lambda(fn.roundCode, input_shape=(2*k,), output_shape=(2*k,), name='Codeword'),
        ], name='Encoder')

NoiseL = tf.keras.Sequential([
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(2*k,), output_shape=(2*k,), name='Noise'),
        ], name='Noise')

Decoder1H = tf.keras.Sequential([ # Array to define layers
        layers.Dense(128, activation='relu', input_shape=(2*k,), name='DHL3'),
        # Add layer with 2**k output units: onehot output
        layers.Dense(256, activation='softmax', name='1H_Output')
        ], name = 'Decoder')

Autoencoder1H = tf.keras.Sequential([Encoder,NoiseL, Decoder1H])
plot_model(Autoencoder1H,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
Autoencoder1H.compile(loss=lossFunc ,
              optimizer='adam',
              )
'''
    Summaries and checkpoints (to do)
'''
summary = Autoencoder1H.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Checkpoints/'+timestr+'_'+title+'_weights.{epoch:02d}-{loss:.6f}.hdf5', monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=2**11)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = Autoencoder1H.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)

# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
#plt.plot(history.history['metricBER'])
plt.grid(True, which='both')
#plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss', 'BER'])
plt.show()
trainingFig.set_size_inches(width, height)
trainingFig.savefig('training_history/'+title+'/'+timestr + '_'+title+'_train.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
Autoencoder1H.save('Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%%
'''
    Prediction 1H
'''
globalReps = 100
globalErrorAutoencoder1H = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = Encoder.predict(u)
        xround = np.round(x)
        xflat = np.reshape(xround, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,2*k) # noisy codewords
        prediction = Decoder1H.predict(y)
        predictedMessages = fn.multipleOneshot2messages(prediction, messages)

        globalErrorAutoencoder1H[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder1H, 'One-hot Autoencoder')