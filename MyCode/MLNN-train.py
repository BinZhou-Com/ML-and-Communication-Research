# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:15:35 2019

@author: user
"""
#%%
u_train_labels = messages.copy()
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.generteCodeWord(test_Size, n, u_val_labels, G)

'''
    Sequential Model: most simple tf MLNN model
'''
NoiseL = tf.keras.Sequential([
        # Noise Layer
        layers.Lambda(tensorBSC,input_shape=(n,), output_shape=(n,), name='Noise'),
        ], name='Noise')
MLNNDecoder = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(128, activation='relu', input_shape=(n,), name='HL1'),
              # Add another: L2
              layers.Dense(64, activation='relu', name='HL2'),
              # Add another: L3
              layers.Dense(32, activation='relu', name='HL3'),
              # Add layer with k output units:
              layers.Dense(k, activation='sigmoid', name='Output')
              ], name='Array_Decoder')
MLNN = tf.keras.Sequential([NoiseL, MLNNDecoder])
plot_model(MLNN,to_file='GraphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
MLNN.compile(loss=lossFunc ,
              optimizer='adam')
'''
    Summaries and checkpoints 
'''
summary = MLNN.summary()
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
#plt.plot(history.history['metricBER'])
plt.grid(True, which='both')
#plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss'])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
trainingFig.savefig('training_history/'+title+'/'+timestr + '_'+title+'_train.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
MLNN.save(path)  # creates a HDF5 file
