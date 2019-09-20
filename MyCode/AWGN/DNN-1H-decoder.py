# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:30:19 2019

@author: user
"""
'''
    DNN One Hot Model Decoder, Noise and Hidden layers separated
'''
'''
    Training and validation data
'''
# Training parameters
#title = 'NN1H'
timestr = time.strftime("%Y%m%d-%H%M%S")
#elevado = 20
#numEpochs = 2**elevado
#batchSize = 256

u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = possibleRealCodewords.copy()

'''
    Architecture
'''
#layerWidth = [128,64,32,256]
#lw = str(layerWidth).replace(" ", "")
train_snr = 1
NoiseL = tf.keras.Sequential([
        # Noise Layer
        #layers.Lambda(fn.tensorAWGN,input_shape=(n,), output_shape=(n,), name='Noise'),
        layers.GaussianNoise(stddev=np.sqrt(1/(2*train_snr)), input_shape=(n,))
        ], name='Noise')
NN1HDecoder = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(layerWidth[0], activation='relu', input_shape=(n,), name='HL1'),
              # Add another: L2
              layers.Dense(layerWidth[1], activation='relu', name='HL2'),
              # Add another: L3
              layers.Dense(layerWidth[2], activation='relu', name='HL3'),
              # Add layer with k output units:
              layers.Dense(layerWidth[3], activation='softmax', name='1H_Output')
              ], name='1H_Decoder')
NN1H = tf.keras.Sequential([NoiseL, NN1HDecoder])
plot_model(NN1H,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)

'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
NN1H.compile(loss=lossFunc ,
              optimizer='adam')
'''
    Summaries and checkpoints 
'''
summary = NN1H.summary()
callbacks_list = []
''' 
    Training
'''
history = NN1H.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)
#history = NN1H.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
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

trainingFig.savefig('GraphNN/'+title+'/'+title+'_train'+ timestr+'.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
NN1H.save('Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'.h5')  # creates a HDF5 file


'''
    Prediction
'''
Decoder=NN1H
fn.NN1HSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages)

'''
    Ploting
'''
filename = './Data/NN1H/NN1H_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
with open(filename, 'rb') as f:
    avgNN1HError = pickle.load(f)
markerlist = ['', '^']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgNN1HError], 
            ['Soft MAP Decoder', '$M_{ep} = 2^{'+str(elevado)+'}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/NN1H_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)


