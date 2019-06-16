# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:15:26 2019

@author: user
"""
#%%
'''
    DNN One Hot Model Decoder
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
    Sequential Model: most simple tf MLNN model
'''

MLNN1H = tf.keras.Sequential([ # Array to define layers
              #layers.Dense(32, activation='relu', input_shape=(n,), name='HL1'),
              # Add layer with 2**k output units:
              layers.Dense(256, activation='softmax', input_shape=(n,),name='Output')
])

plot_model(MLNN1H,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
MLNN1H.compile(loss=lossFunc ,
              optimizer='adam')
'''
    Summaries and checkpoints (to do)
'''
summary = MLNN1H.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Checkpoints/'+timestr+'_'+title+'_weights.{epoch:02d}-{loss:.6f}.hdf5', monitor='loss', 
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
plotTraining(history)

'''
    Saving model
'''

MLNN1H.save(path)  # creates a HDF5 file