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

u_train_labels = messages.copy()
x_train_data = u_train_labels

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

encoderNodes = np.array([64, 128, 256, 16])
DecoderNodes = [128, 64, 32, k]
#%
'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(k,), name='Input'),
        # Hidden Layer
        layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        # Hidden Layer
        layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        # Coded Layer
        layers.Dense(encoderNodes[3], activation='sigmoid', name='Codedfloat'),
        # Rounded codeword
        layers.Lambda(fn.roundCode, input_shape=(encoderNodes[3],), 
                      output_shape=(encoderNodes[3],), name='Codeword'),
        ], name='Encoder')

NoiseL = tf.keras.Sequential([
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(encoderNodes[3],), 
                      output_shape=(encoderNodes[3],), name='Noise'),
        ], name='Noise')

Decoder = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        layers.Dense(DecoderNodes[0], activation='relu', input_shape=(encoderNodes[3],), name='DHL1'),
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

#%
'''
    Prediction Array
'''
'''
globalReps = 1000
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = Encoder.predict(u)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,encoderNodes[3]) # noisy codewords
        prediction = Decoder.predict(y)
        predictedMessages = np.round(prediction)
        globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder, 'Array Autoencoder')
'''




