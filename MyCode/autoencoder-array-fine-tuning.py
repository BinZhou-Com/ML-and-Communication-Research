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



