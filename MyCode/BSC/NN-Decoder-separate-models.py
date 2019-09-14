# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:04:54 2019

@author: user
"""
#%%
'''
    DNN One Hot Model Decoder, Noise and Hidden layers separated
'''
'''
    Training and validation data
'''
u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)
u_val_labels = fn.messages2onehot(u_val_labels)

'''
    Constants
'''
numEpochs = 2**11  #2**16 approx 65000
batchSize = trainSize 
train_p = 0.07
timestr = time.strftime("%Y%m%d-%H%M%S")

'''
    Architecture
'''
NoiseL = tf.keras.Sequential([
            # Noise Layer
            layers.Lambda(tensorBSC,input_shape=(n,), output_shape=(n,), name='Noise'),
        ])
Decoder1H = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(32, activation='relu', input_shape=(n,), name='HL1'),
              # Add another: L2
              layers.Dense(64, activation='relu', name='HL2'),
              # Add another: L3
              layers.Dense(128, activation='relu', name='HL3'),
              # Add layer with k output units: onehot output
              layers.Dense(256, activation='softmax', name='1H_Output')
])
MergedModel = tf.keras.Sequential([NoiseL, Decoder1H], name=['Noise', '1HNNDecoder'])
plot_model(MergedModel,to_file='graphNN/merged1H/'+timestr+'_merged1H.png',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
MergedModel.compile(loss=lossFunc ,
              optimizer='adam',
              metrics=[fn.metricBER1H])
'''
    Summaries and checkpoints (to do)
'''
summary = MergedModel.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Checkpoints/'+timestr+'_merged1H_weights.{epoch:02d}-{loss:.6f}.hdf5', monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=2**11)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = MergedModel.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)
#history = MergedModel.fit(x_train_data, u_train_labels, epochs=numEpochs, 
#                           batch_size=batchSize, shuffle=True, verbose=0,
#                           validation_data=(x_val, u_val_labels))

# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
plt.plot(history.history['metricBER1H'])
plt.grid(True, which='both')
plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss', 'BER'])
plt.show()
trainingFig.set_size_inches(width, height)
trainingFig.savefig('training_history/'+timestr + '_merged1H_train.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
MergedModel.save('Trained_NN_1H/'+timestr+'_merged1H_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5')  # creates a HDF5 file

#%%
'''
    Prediction
'''
globalReps = 1000
globalErrorMerged1H = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        prediction = Decoder1H.predict(y)
        predictedMessages = fn.multipleOneshot2messages(prediction, messages)

        globalErrorMerged1H[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)

#% Plotting
avgGlobalErrorMerged1H = np.average(globalErrorMerged1H,0)

fig = plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(pOptions,avgGlobalErrorMerged1H, color='g', marker='^', zorder=3)
plt.plot(pOptions,avgGlobalError, color='b')
plt.plot(pOptions,avgGlobalErrorMAP, color='r')
plt.grid(True, which='both')
#plt.title('Batch size = '+str(batchSize)+', train_p = ' + str(train_p))
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'MAP', 'DNN Decoder, $M_{ep}=$'+str(numEpochs)])
plt.show()

#timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('images/'+timestr+'_MAP_merged1H_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)
