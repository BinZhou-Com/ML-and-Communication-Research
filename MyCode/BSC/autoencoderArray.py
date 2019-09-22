# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:01:18 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:04:54 2019

@author: user
"""
#%%
'''
    Array training and validation data
'''
timestr = time.strftime("%Y%m%d-%H%M%S")
title='AutoencoderArray'
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'

u_train_labels = messages.copy()
x_train_data = u_train_labels

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

#encoderNodes = np.array([512, 256]) - best so far
encoderNodes = np.array([512, 256, 256])
DecoderNodes = [128, 64, 32]
#%
'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(k,), name='Input'),
        #layers.Dropout(rate=0.1),
        # Hidden Layer
        layers.BatchNormalization(),
        layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        layers.BatchNormalization(),
        #layers.Dropout(rate=0.1), 
        # Hidden Layer
        #layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        #layers.BatchNormalization(),
        # Coded Layer
        layers.Dense(n, activation='sigmoid', name='Codedfloat')
        ], name='Encoder')

NoiseL = tf.keras.Sequential([
        # Rounded codeword
        layers.Lambda(roundCode, input_shape=(n,), 
                      output_shape=(n,), name='Codeword'),
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(n,), 
                      output_shape=(n,), name='Noise'),
        ], name='Noise')

Decoder = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        layers.Dense(DecoderNodes[0], activation='relu', input_shape=(n,), name='DHL1'),
        # Add another: L2
        layers.Dense(DecoderNodes[1], activation='relu', name='DHL2'),
        # Add another: L3
        layers.Dense(DecoderNodes[2], activation='relu', name='DHL3'),
        # Add layer with k output units: output
        layers.Dense(DecoderNodes[3], activation='sigmoid', name='Output')
        ], name = 'Decoder')

Autoencoder = tf.keras.Sequential([Encoder,NoiseL, Decoder])
plot_model(Autoencoder,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilationl
'''
lossFunc = 'logcosh'
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

es =  tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000000001, patience=256, verbose=1, mode='auto')          
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
'''
    Prediction Array

globalReps = 1000
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = Encoder.predict(u)
        x = np.round(x)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        prediction = Decoder.predict(y)
        predictedMessages = np.round(prediction)
        globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder, 'Array Autoencoder')
'''




