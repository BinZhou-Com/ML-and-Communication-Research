# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:31:47 2019

@author: user
"""
'''
    Array Autoencoder
'''
# Training parameters
#title = 'AAutoencoder'
timestr = time.strftime("%Y%m%d-%H%M%S")
#elevado = 12
#numEpochs = 2**elevado
batchSize = 256

u_train_labels = fn.BPSK(messages.copy())
x_train_data = messages.copy()

trainSize = np.size(x_train_data, 0)



'''
    Sequential Model: most simple tf Autoencoder model
'''
#encoderNodes = [32, 64, 128, 16]
#decoderNodes = [128, 64, 32, 8]
#lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
#train_snr = 1

Encoder = tf.keras.Sequential([
        # Input Layer
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(k,), name='Input'),
        #layers.Dropout(rate=0.1),
        # Hidden Layer
        #layers.BatchNormalization(),
        layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        #layers.BatchNormalization(),
        #layers.Dropout(rate=0.1), 
        # Hidden Layer
        #layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        #layers.BatchNormalization(),
        # Coded Layer
        layers.Dense(encoderNodes[3], activation='linear', name='Codedfloat'),
        layers.BatchNormalization(),
        layers.Lambda(fn.normalize, input_shape=(n,), output_shape=(n,)),
        
        ], name='Array_Encoder')

NoiseL = tf.keras.Sequential([
        # Noise Layer
        #layers.Lambda(fn.tensorAWGN,input_shape=(n,), output_shape=(n,), name='Noise'),
        layers.GaussianNoise(stddev=np.sqrt(1/(2*train_snr)), 
                             input_shape=(n,))
        ], name='Noise')
Decoder = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(decoderNodes[0], activation='relu', input_shape=(n,), name='DHL1'),
              # Add another: L2
              layers.Dense(decoderNodes[1], activation='relu', name='DHL2'),
              # Add another: L3
              layers.Dense(decoderNodes[2], activation='relu', name='DHL3'),
              # Add layer with k output units:
              layers.Dense(decoderNodes[3], activation='sigmoid', name='Output')
              ], name='Array_Decoder')
AAutoencoder = tf.keras.Sequential([Encoder, NoiseL, Decoder])
plot_model(AAutoencoder,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'mean_squared_error'
AAutoencoder.compile(loss=lossFunc,
              optimizer='adam')
'''
    Summaries and checkpoints 
'''
summary = AAutoencoder.summary()
callbacks_list = []
''' 
    Training
'''
history = AAutoencoder.fit(x_train_data, u_train_labels, epochs=numEpochs, 
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

trainingFig.savefig('GraphNN/'+title+'/'+title+'_train'+ timestr+'.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
AAutoencoder.save('Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'.h5')  # creates a HDF5 file


'''
    Prediction
'''
fn.AutoencoderSinglePredictionAWGN(Encoder, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title)

'''
    Ploting
'''
filename = './Data/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
with open(filename, 'rb') as f:
    avgAAutoencoderError = pickle.load(f)
markerlist = ['', 'o']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgAAutoencoderError], 
            ['Soft MAP Decoder', 'Array Autoencoder'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
#plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/AAutoencoder_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)
