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

encoderNodes = np.array([196, 128, 96, 64, 32]) # add a 192 layer at the beginning
DecoderNodes = [128, 192]
#%
'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(256,), name='Input'),
        layers.BatchNormalization(),
        # Hidden Layer
        layers.Dense(encoderNodes[1], activation='relu', name='EHL'),
        layers.BatchNormalization(),
        # Hidden Layer
        layers.Dense(encoderNodes[2], activation='relu', name='EHL1'),
        layers.BatchNormalization(),
        # Hidden Layer
        layers.Dense(encoderNodes[3], activation='relu', name='EHL2'),
        layers.BatchNormalization(),
        # Hidden Layer
        layers.Dense(encoderNodes[4], activation='relu', name='EHL3'),
        layers.BatchNormalization(),
        # Coded Layer
        layers.Dense(n, activation='sigmoid', name='Codedfloat'),
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
        layers.Dense(DecoderNodes[0], activation='relu', input_shape=(n,), name='DHL1'),
        layers.Dense(256, activation='softmax', name='1H_Output')
        ], name = 'Decoder')

Autoencoder1H = tf.keras.Sequential([Encoder,NoiseL, Decoder])
plot_model(Autoencoder1H,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'logcosh'
Autoencoder1H.compile(loss=lossFunc ,
              optimizer='adam'
              )
#metrics=[fn.metricBER1H]
'''
    Summaries and checkpoints (to do)
'''
summary = Autoencoder1H.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpointPath, monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=checkpointPeriod)
            
es =  tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000000001, patience=256, verbose=1, mode='auto')          
#es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1024)
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
Autoencoder1H.save(path)  # creates a HDF5 file

#%%%
'''
    Prediction 1H
'''
t = TicToc('name')
t.tic()


globalReps = 100
globalErrorAutoencoder1H = fn.onehotAutoencoderPrediction(Encoder, Decoder, 
                               messages, pOptions, globalReps, N, n, k)

#% Plotting
plotBERp(globalErrorAutoencoder1H, 'One-hot Autoencoder')

t.toc()
print(t.elapsed)

