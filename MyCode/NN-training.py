# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:12:38 2019

@author: user
"""
#%% Functions
def tensorBSC(x):
    # value of p: optimal training statistics for neural based channel decoders (paper)
    p = K.constant(train_p,dtype=tf.float32)
    var = K.random_uniform(shape=(func_output_shape(x),), minval = 0.0, maxval=1.0)
    noise = K.less(var, p)
    noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noiseFloat, x)%2
    return result
    
#%% Neural Networ decoder
'''
    DNN ARRAY Decoder
'''
'''
    Training and validation data
'''
n=16
k=8

u_train_labels = messages.copy()
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)


'''
    Array Decodingg
'''
'''
    Constants
'''

numEpochs = 2**14  #2**16 approx 65000
batchSize = 256 # Mini batch size
train_p = 0.07
    
'''
    Sequential Model: most simple tf MLNN model
'''
MLNN = tf.keras.Sequential([ # Array to define layers
              # Noise Layer
              layers.Lambda(fn.tensorBSC(train_p),input_shape=(n,), output_shape=(n,)),
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(128, activation='relu', input_shape=(n,)),
              # Add another: L2
              layers.Dense(64, activation='relu'),
              # Add another: L3
              layers.Dense(32, activation='relu'),
              # Add layer with k output units:
              layers.Dense(k, activation='sigmoid')
])
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'mse'
MLNN.compile(loss=lossFunc ,
              optimizer='adam',
              metrics=[fn.metricBER])
'''
    Summaries and checkpoints 
'''
callbacks_list = []
''' 
    Training
'''
history = MLNN.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)
#history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
#          validation_data=(x_val, u_val_labels))

# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
plt.plot(history.history['metricBER'])
plt.grid(True, which='both')
#plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss', 'BER'])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
trainingFig.savefig('training_history/'+timestr + '_train.png', bbox_inches='tight')
'''
    evaluate the inference-model
''' 

evaluation = MLNN.evaluate(x_val, u_val_labels)

'''
    Saving model
'''
MLNN.save('Trained_NN/'+timestr+'MLNN_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%%
'''
    Prediction
'''
globalReps = 100
globalErrorMLNN = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        prediction = MLNN.predict(y)
        # round predictions
        rounded = np.round(prediction)

        globalErrorMLNN[i_global][i_p] = fn.bitErrorFunction(rounded, u)


#%%     
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)

fig = plt.figure(figsize=(8, 6), dpi=80)

plt.plot(pOptions,avgGlobalError, color='b')
plt.plot(pOptions,avgGlobalErrorMAP, color='r')

avgGlobalErrorMLNN = np.average(globalErrorMLNN,0)
plt.scatter(pOptions,avgGlobalErrorMLNN, color='g')

plt.grid(True, which='both')
plt.title('Batch size = '+str(batchSize))
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP', 'DNN Decoder, $M_{ep}=$'+str(numEpochs)])
plt.show()

fig.savefig('images/'+timestr+'MAP_MLNN_Mep_'+str(numEpochs)+'.png', bbox_inches='tight')

#%% One hot training

'''
    DNN One Hot Model Decoder
'''
'''
    Training and validation data
'''
u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 8, axis=0)
x_train_data = np.repeat(x_train_data, 8, axis=0)
trainSize = np.size(x_train_data, 0)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)
u_val_labels = fn.messages2onehot(u_val_labels)

'''
    Constants
'''
numEpochs = 2**11  #2**16 approx 65000
#batchSize = trainSize 
batchSize = 256  
train_p = 0.0
timestr = time.strftime("%Y%m%d-%H%M%S")
'''
    Sequential Model: most simple tf MLNN model
'''
MLNN1H = tf.keras.Sequential([ # Array to define layers
              # Noise Layer
              layers.Lambda(fn.tensorBSC,input_shape=(n,), output_shape=(n,)),
              # Adds a densely-connected layer with n units to the model: L1
              #layers.Dense(32, activation='relu', input_shape=(n,)),
              # Add another: L2
              layers.Dense(64, activation='relu'),
              # Add another: L3
              layers.Dense(128, activation='relu'),
              # Add layer with k output units:
              layers.Dense(256, activation='softmax')
])
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
#lossFunc = 'mse'
MLNN1H.compile(loss=lossFunc ,
              optimizer='adam',
              metrics=[fn.metricBER1H])
'''
    Summaries and checkpoints (to do)
'''
summary = MLNN1H.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Checkpoints/'+timestr+'weights.{epoch:02d}-{loss:.6f}.hdf5', monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=2**11)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = MLNN1H.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)
#history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
#          validation_data=(x_val, u_val_labels))

# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
plt.plot(history.history['metricBER1H'])
plt.grid(True, which='both')
#plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss', 'BER'])
plt.show()

trainingFig.savefig('training_history/'+timestr + '_train.png', bbox_inches='tight')
'''
    evaluate the inference-model
''' 

evaluation = MLNN1H.evaluate(x_val, u_val_labels)

u = fn.generateU(1,k)
y = fn.generteCodeWord(1, n, u, G)
prediction = MLNN1H.predict(y)
predictedMessage = fn.onehot2singleMessage(prediction, messages)

'''
    Saving model
'''
MLNN1H.save('Trained_NN_1H/'+timestr+'MLNN1H_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%%
'''
    Prediction
'''
globalReps = 1000
globalErrorMLNN1H = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        prediction = MLNN1H.predict(y)
        predictedMessages = fn.multipleOneshot2messages(prediction, messages)

        globalErrorMLNN1H[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)
avgGlobalErrorMLNN1H = np.average(globalErrorMLNN1H,0)

fig = plt.figure(figsize=(8, 6), dpi=80)

plt.plot(pOptions,avgGlobalError, color='b')
plt.plot(pOptions,avgGlobalErrorMAP, color='r')
plt.scatter(pOptions,avgGlobalErrorMLNN1H, color='g')

plt.grid(True, which='both')
plt.title('Batch size = '+str(batchSize)+', train_p = ' + str(train_p))
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP', 'DNN Decoder, $M_{ep}=$'+str(numEpochs)])
plt.show()

#timestr = time.strftime("%Y%m%d-%H%M%S")
fig.savefig('images/'+timestr+'MAP_MLNN1H_Mep_'+str(numEpochs)+'.png', bbox_inches='tight')