# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:01:59 2019

@author: user
"""
'''
    Array Decoder
'''
# Training parameters
#title = 'MLNN'
#timestr = time.strftime("%Y%m%d-%H%M%S")
#numEpochs = 2**14
batchSize = 256

u_train_labels = messages.copy()
x_train_data = possibleRealCodewords.copy()

#u_train_labels = np.repeat(u_train_labels, 1, axis=0)
#x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

test_Size = 100
u_val_labels = fn.generateU(test_Size,k)
x_val = fn.BPSK(fn.generteCodeWord(test_Size, n, u_val_labels, G))

'''
    Sequential Model: most simple tf MLNN model
'''
#layerWidth = [128,64,32,k]
#lw = str(layerWidth).replace(" ", "")
train_snr = 1
NoiseL = tf.keras.Sequential([
        # Noise Layer
        #layers.Lambda(fn.tensorAWGN,input_shape=(n,), output_shape=(n,), name='Noise'),
        layers.GaussianNoise(stddev=np.sqrt(1/(2*train_snr)), input_shape=(n,))
        ], name='Noise')
MLNNDecoder = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(layerWidth[0], activation='relu', input_shape=(n,), name='HL1'),
              # Add another: L2
              layers.Dense(layerWidth[1], activation='relu', name='HL2'),
              # Add another: L3
              layers.Dense(layerWidth[2], activation='relu', name='HL3'),
              # Add layer with k output units:
              layers.Dense(layerWidth[3], activation='sigmoid', name='Output')
              ], name='Array_Decoder')
MLNN = tf.keras.Sequential([NoiseL, MLNNDecoder])
plot_model(MLNN,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)
    
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

trainingFig.savefig('GraphNN/'+title+'/'+title+'_train'+ timestr+'.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
MLNN.save('Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'.h5')  # creates a HDF5 file


'''
    Prediction
'''
Decoder=MLNN
fn.MLNNSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title)

'''
    Ploting
'''
filename = './Data/MLNN/MLNN_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
with open(filename, 'rb') as f:
    avgMLNNError = pickle.load(f)
markerlist = ['', '^']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgMLNNError], 
            ['Soft MAP Decoder', '$M_{ep} = 2^{'+str(elevado)+'}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MLNN_vs_Decoder_'+lw+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)