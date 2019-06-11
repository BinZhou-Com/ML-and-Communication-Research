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
x_train_data = u_train_labels.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

encoderNodes = np.array([256, 256, 256])
DecoderNodes = [256, 256]
#%
'''
    Constants
'''
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'
'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(256,), name='Input'),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
            scale=True, beta_initializer='zeros', gamma_initializer='ones', 
            moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        # Hidden Layer
        layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
            scale=True, beta_initializer='zeros', gamma_initializer='ones', 
            moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        # Hidden Layer
        #layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        # Coded Layer
        layers.Dense(n, activation='sigmoid', name='Codedfloat'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        ], name='Encoder')

NoiseL = tf.keras.Sequential([
        # Rounded codeword
        layers.Lambda(fn.roundCode, input_shape=(n,), 
                      output_shape=(n,), name='Codeword'),
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(n,), 
                      output_shape=(n,), name='Noise'),
        ], name='Noise')

Decoder1H = tf.keras.Sequential([ # Array to define layers
        layers.Dense(DecoderNodes[0], activation='relu', input_shape=(n,), name='DHL1'),
        layers.Dense(DecoderNodes[1], activation='relu', name='DHL2'),
        layers.Dense(256, activation='softmax', name='1H_Output')
        ], name = 'Decoder')

Autoencoder1H = tf.keras.Sequential([Encoder,NoiseL, Decoder1H])
plot_model(Autoencoder1H,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'mse'
Autoencoder1H.compile(loss=lossFunc ,
              optimizer='adam',
              )
'''
    Summaries and checkpoints (to do)
'''
summary = Autoencoder1H.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpointPath, monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=checkpointPeriod)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = Autoencoder1H.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)

# summarize history for loss
plotTraining(history)

'''
    Saving model
'''
Autoencoder1H.save('Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%
'''
    Prediction 1H
'''
t = TicToc('name')
t.tic()


globalReps = 100
globalErrorAutoencoder1H = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        u1h = fn.messages2onehot(u)
        x = Encoder.predict(u1h)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,2*k) # noisy codewords
        prediction = Decoder1H.predict(y)
        predictedMessages = fn.multipleOneshot2messages(prediction, messages)

        globalErrorAutoencoder1H[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder1H, 'One-hot Autoencoder')

t.toc()
print(t.elapsed)
