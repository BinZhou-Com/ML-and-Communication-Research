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
numEpochs = 2**16  #2**16 approx 65000
batchSize = trainSize 
train_p = 0.07
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'Autoencoder'

'''
    Architecture
'''
def roundCode(x):
    return tf.stop_gradient(K.round(x)-x)+x
    #return K.round(x)

def skipLayer(x):
    return x + K.stop_gradient(x)

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
        #layers.Lambda(lambda x: K.stop_gradient(x)),
        layers.Lambda(roundCode, input_shape=(2*k,), output_shape=(2*k,), name='Codeword', trainable=False),
        layers.Lambda(lambda x: K.stop_gradient(x), output_shape=(2*k,))
        #layers.Lambda(lambda x: K.stop_gradient(x)), # Stop gradient
        ], name='Encoder')
'''
inputL = layers.Input(shape=(k,), name='Input')
HL1 = layers.Dense(128, activation='relu', name='HL1')(inputL)
HL2 = layers.Dense(64, activation='relu', name='HL2')(HL1)
HL3 = layers.Dense(32, activation='relu', name='HL3')(HL2)
HL4 = layers.Dense(2*k, activation='sigmoid', name='CodedFloat')(HL3)
roundedL = layers.Lambda(roundCode, output_shape=(2*k,), name='Codeword')(HL4)
#stop_grad = layers.Lambda(skipLayer, output_shape=(2*k,))(roundedL)
#stop_grad = layers.Lambda(lambda x: K.stop_gradient(x))(roundedL)
Encoder = tf.keras.Model(inputs=inputL, outputs=roundedL, name='Encoder')

plot_model(Encoder,to_file='graphNN/'+title+'/'+timestr+'_'+title+'_Encoder.pdf',show_shapes=True)

EncoderS = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        layers.Dense(128, activation='relu', input_shape=(k,), name='EHL1'),
        # Add another: L2
        layers.Dense(64, activation='relu', name='EHL2'),
        # Add another: L3
        layers.Dense(32, activation='relu', name='EHL3'),
        # Add layer with k output units: onehot output
        layers.Dense(2*k, activation='sigmoid', name='Output')
        ], name = 'Encoder')
NoiseL = tf.keras.Sequential([
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(2*k,), output_shape=(2*k,), name='Noise'),
        ], name='Noise')
Decoder1H = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        layers.Dense(32, activation='relu', input_shape=(2*k,), name='DHL1'),
        # Add another: L2
        layers.Dense(64, activation='relu', name='DHL2'),
        # Add another: L3
        layers.Dense(128, activation='relu', name='DHL3'),
        # Add layer with 2**k output units: onehot output
        layers.Dense(256, activation='softmax', name='1H_Output')
        ], name = 'Decoder')
Decoder = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        layers.Dense(32, activation='relu', input_shape=(2*k,), name='DHL1'),
        # Add another: L2
        layers.Dense(64, activation='relu', name='DHL2'),
        # Add another: L3
        layers.Dense(128, activation='relu', name='DHL3'),
        # Add layer with k output units: output
        layers.Dense(k, activation='sigmoid', name='Output')
        ], name = 'Decoder')
#Autoencoder = tf.keras.Sequential([EncoderS, NoiseL, Decoder1H])
Autoencoder = tf.keras.Sequential([Encoder,NoiseL, Decoder])
plot_model(Autoencoder,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
Autoencoder.compile(loss=lossFunc ,
              optimizer='adam',
              )
'''
    Summaries and checkpoints (to do)
'''
summary = Autoencoder.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Checkpoints/'+timestr+'_'+title+'_weights.{epoch:02d}-{loss:.6f}.hdf5', monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=2**11)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = Autoencoder.fit(x_train_data, u_train_labels, epochs=numEpochs, 
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
Autoencoder.save('Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%%
'''
    Prediction 1H
'''
globalReps = 100
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
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

        globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder, 'Autoencoder')

#%%
'''
    Prediction Array
'''
globalReps = 100
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = Encoder.predict(u)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,2*k) # noisy codewords
        prediction = Decoder.predict(y)
        predictedMessages = np.round(prediction)
        globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
plotBERp(globalErrorAutoencoder, 'Autoencoder')


