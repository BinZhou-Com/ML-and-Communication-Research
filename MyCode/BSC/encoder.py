# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:26:59 2019

@author: user
"""
#%%
'''
    Array training and validation data
'''
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'

u_train_labels = possibleCodewords.copy()
x_train_data = fn.messages2onehot(messages)

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

encoderNodes = np.array([32, 32])

'''
    Architecture
'''
Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(256,), name='Input'),
        #layers.Conv1D(12, kernel_size= 4, activation='relu', input_shape=(k,), name='Input'),
        # Dropout Layer
        #layers.Dropout(rate=0.01),
        # Hidden Layer
        layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        # Hidden Layer
        #layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        # Hidden Layer
        #layers.Dense(encoderNodes[3], activation='relu', name='EHL3'),
        # Coded Layer
        layers.Dense(n, activation='sigmoid', name='Codedfloat')
        # Normalization Layer
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')
        ], name='Encoder')

trainingPath = 'training_history/'+title+'/'+timestr + '_'+title+'_train_' + str(encoderNodes) +'_1H.png'
plot_model(Encoder,to_file='graphNN/'+title+'/'+timestr+'_'+title+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilationl
'''
lossFunc = 'mse'
Encoder.compile(loss=lossFunc ,
              optimizer='adam',
              )
'''
    Summaries and checkpoints (to do)
'''
summary = Encoder.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpointPath, monitor='loss', 
        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=checkpointPeriod)
callbacks_list = [checkpoint]
''' 
    Training
'''
history = Encoder.fit(x_train_data,u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)

plotTraining(history)
'''
    Saving model
'''
Encoder.save(path)  # creates a HDF5 file

#%%
'''
    Prediction Array

globalReps = 100
globalErrorEncoder = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        xhat = np.round(Encoder.predict(u))
        
        globalErrorEncoder[i_global][i_p] = fn.codeErrorFunction(xhat, x)

#% Plotting
plotBERp(globalErrorEncoder, 'Encoder')
'''
#%%
'''
    Prediction MAP with NN encoder
'''
globalReps = 100
globalErrorMAP2 = np.empty([globalReps, len(pOptions)])
possibleCodewords = np.round(fn.messages2customEncoding(messages, Encoder))

for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = np.round(fn.messages2customEncoding(u, Encoder))
       
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        
        MAP2 = np.empty([N,k])
        for i in range(N):
            minDistWord = np.argmin(np.sum(possibleCodewords!=y[i], 1), 0) # find word of minimum distance
            MAP2[i] = messages[minDistWord]
            
        globalErrorMAP2[i_global][i_p] = fn.bitErrorFunction(MAP2, u)

#% Plotting
plotBERp(globalErrorMAP2, 'MAP with NN Encoder')

#%%
'''
    Predicition one-hot encoder
'''
globalReps = 100
globalErrorEncoder = np.empty([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        u1h = fn.messages2onehot(u)
        xhat = np.round(Encoder.predict(u1h))
        
        globalErrorEncoder[i_global][i_p] = fn.codeErrorFunction(xhat, x)

#% Plotting
plotBERp(globalErrorEncoder, 'Encoder')

