# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:30:24 2019

@author: user
"""
#%%
'''
    Constants
'''
numEpochs = 2**13  #2**16 approx 65000
batchSize = 256
train_p = 0.0
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'Decoder-fine-tuning'
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'

'''
    Load Encoder model
'''
directory = 'Fine-tuning/decoder/'
fileName = 'AutoencoderArray_Mep_262144_p_0.07.h5'
loadedModel = tf.keras.models.load_model(directory+fileName)
Decoder = loadedModel
print("Loaded model from disk")

#Encoder = loadedModel.layers[0]
# Save encoder separately
#Encoder.save('Fine-tuning/encoder/'+fileName)  # creates a HDF5 file

'''
    Array training and validation data
'''
timestr = time.strftime("%Y%m%d-%H%M%S")

u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = fn.messages2customEncoding(messages, Encoder)

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

#%% Initialization
MAPpredictTime = TicToc('MAPPredict')
globalErrorMAP2 = np.empty([globalReps, len(pOptions)])
title = 'MAP'
possibleCodewords = fn.messages2customEncoding(messages, Encoder)
MAPpredictTime.tic()
for i_global in range(globalReps):
    for i_p in range(np.size(pOptions)):
        p = pOptions[i_p]
        
        '''
               Generate channel Input
        '''
        
        u = fn.generateU(N,k)
        x = fn.messages2customEncoding(u, Encoder)
        
        ''' 
            Channel
        '''
        
        xflat = np.reshape(x, [-1])
        yflat = fn.BSC(xflat,p)
        y = yflat.reshape(N,n) # noisy codewords
        
        '''
            Decoding
        '''
        '''
            MAP Decoder (minimum distance decoding)
        '''

        MAP = np.empty([N,k])
        for i in range(N):
            minDistWord = np.argmin(np.sum(possibleCodewords!=y[i], 1), 0) # find word of minimum distance
            MAP[i] = messages[minDistWord]
        
        '''
            Error Calculation
        '''
        
        globalErrorMAP2[i_global][i_p] = fn.bitErrorFunction(MAP, u)

MAPpredictTime.toc()
print('Total MAP predict time: ', MAPpredictTime.elapsed)
#% plot
        
avgGlobalError = np.average(globalError, 0)
avgGlobalErrorMAP = np.average(globalErrorMAP, 0)
avgGlobalErrorMAP2 = np.average(globalErrorMAP2, 0)

fig = plt.figure(figsize=(8, 6), dpi=80)

plt.plot(pOptions,avgGlobalError, color='b', linewidth=lineWidth)
plt.plot(pOptions,avgGlobalErrorMAP, color='r', linewidth=lineWidth)
plt.plot(pOptions,avgGlobalErrorMAP2, color='g', linewidth=lineWidth)

plt.grid(True, which="both")
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(['No Decoding', 'Hamming MAP', 'DNN Encoder MAP'])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('images/'+title+'/'+timestr+'_MAP.png', bbox_inches='tight', dpi=300)

#%% 
'''
    Compare loaded autoencoder models
'''
p_trainOptions = [0.03, 0.03]
directory = 'Saved_Models\AutoencoderArray\p_train_03\\'
fileName = ['i_1_AutoencoderArray_Mep_65536_p_0.03.h5', 'i_2_AutoencoderArray_Mep_65536_p_0.03.h5']
Models = [tf.keras.models.load_model(directory+fileName[0]), tf.keras.models.load_model(directory+fileName[1])]
print("Loaded models from disk")

Encoders = [Models[0].layers[0], Models[1].layers[0]]
Decoders = [Models[0].layers[2], Models[1].layers[2]]

'''
    Prediction
'''
globalReps = 1000
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
multiPredictions = np.empty([len(Models),len(pOptions)])
for mod in range(len(Models)):
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = fn.generateU(N,k)
            x = Encoders[mod].predict(u)
            xflat = np.reshape(x, [-1])
            yflat = fn.BSC(xflat,p)
            y = yflat.reshape(N,encoderNodes[3]) # noisy codewords
            prediction = Decoders[mod].predict(y)
            predictedMessages = np.round(prediction)
            globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)
            
    multiPredictions[mod] = np.average(globalErrorAutoencoder,0)

#%%Plotting
fig = plt.figure(figsize=(8, 6), dpi=80)
markers = ['^', 'x', 'o', 's', 'v', '*']

plt.plot(pOptions,avgGlobalError, color='b', linewidth=lineWidth, linestyle='--', label='No Decoding')
plt.plot(pOptions,avgGlobalErrorMAP, color='r', linewidth=lineWidth, label='MAP')
plt.grid(True, which='both')

for i in range(len(Models)):
    plt.scatter(pOptions,multiPredictions[i], marker=markers[i], 
                zorder=3+i, s=markerSize, label='Autoen., $\mathrm{p_t}$ = %s' % p_trainOptions[i])
    
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()

figPath = directory+'\\images'
fn.createDir(figPath)
fig.set_size_inches(width, height)
fig.savefig(figPath+'\MAP_'+title+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)

#%%%
'''
    Check checkedpoints autoencoders
'''
'''
    Constants
'''
numEpochs = 2**14  #2**16 approx 65000
batchSize = 256
train_p = 0.03
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'Autoencoder1H'
path = 'Trained_'+title+'/'+timestr+'_'+title+'_Mep_'+str(numEpochs)+'_bs_'+str(batchSize)+'.h5'
directory = 'Fine-tuning/full-models/'
fileNames = ['20190619-161454_Autoencoder1H_Mep_32768-0.00000000.h5', 
             '20190619-161454_Autoencoder1H_Mep_49152-0.00000000.h5',
             '20190619-161454_Autoencoder1H_Mep_57344-0.00000000.h5',
             '20190619-161454_Autoencoder1H_Mep_69632-0.00000000.h5',
             '20190619-161454_Autoencoder1H_Mep_118784-0.00000000.h5',
             '20190619-161454_Autoencoder1H_Mep_126976-0.00000000.h5']

for i in range(len(fileNames)):  
    '''
        Load model
    '''
    fileName = fileNames[i]
    loadedModel = tf.keras.models.load_model(directory+fileName)
    print("Loaded model from disk")
    
    Encoder = loadedModel.layers[0]
    Decoder = loadedModel.layers[2]
    
    globalReps = 1000
    globalErrorAutoencoder1H = fn.onehotAutoencoderPrediction(Encoder, Decoder, 
                               messages, pOptions, globalReps, N, n, k)
    
    plotBERp(globalErrorAutoencoder1H, 'One-hot Autoencoder')


