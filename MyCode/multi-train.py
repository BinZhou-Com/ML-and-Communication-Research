# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:41:33 2019

@author: user
"""
#%%
'''
    Multi Training
'''
trainTime = TicToc('Training')
trainTimeT = TicToc('TrainingT')
# Constants 
numEpochs = 2**16  #2**16 approx 65000
batchSize = trainSize 
timestr = time.strftime("%Y%m%d-%H%M%S")
title = 'AutoencoderArray'
p_trainOptions = np.array([0.0, 0.01, 0.03, 0.05, 0.07])

myDir = 'Simulation_Mep_64_128_256_65536_bs_256_pmin_0.0_pmax_0.07'
directory = 'Autoencoder_Simulations\\'+ myDir
fn.createDir(directory)
fn.createDir(directory+'\\models')

paths = ["" for x in range(len(p_trainOptions))]
trainTimeT.tic()
for i in range(len(p_trainOptions)):
    trainTime.tic()
    train_p = p_trainOptions[i]
        
    paths[i] = directory+'\\models\\'+str(i)+'.h5'
    path = paths[i]
    checkpointPath = directory+'\\models\\'+'i_'+str(i)+'_'+title+'_p_'+str(train_p)+'_Mep_{epoch:02d}-{loss:.6f}.h5'
    exec(open("autoencoderArray.py").read())
    
    trainTime.toc()
    print('Train time for model '+str(i)+': ',trainTime.elapsed)
trainTimeT.toc()

    
    
#%%
'''
    Multi Prediction
'''
predictTime = TicToc('Predict')
predictTimeT = TicToc('PredictT')
# Initialize variables
multiPredictions = np.empty([len(p_trainOptions),len(pOptions)])
globalReps = 1000
globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
predictTimeT.tic()
for i_train in range(len(p_trainOptions)):
    predictTime.tic()
    # load weights into new model
    fileName = directory + '/models/' + 'i_' +str(i_train) + '_'+title+'_Mep_'+ str(numEpochs) +'_p_'+str(p_trainOptions[i_train])+ '.h5'
    loadedModel = tf.keras.models.load_model(fileName)
    print("Loaded model from disk")
    plot_model(loadedModel, to_file='Debug\Loaded.pdf')
    Encoder = loadedModel.layers[0]
    Decoder = loadedModel.layers[2]
    
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = fn.generateU(N,k)
            #u1h = fn.messages2onehot(u)
            x = np.round(Encoder.predict(u))
            xflat = np.reshape(x, [-1])
            yflat = fn.BSC(xflat,p)
            y = yflat.reshape(N,2*k) # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = np.round(prediction)
        
            globalErrorAutoencoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)
            
    multiPredictions[i_train] = np.average(globalErrorAutoencoder,0)
    #% individual Plotting
    plotBERp(globalErrorAutoencoder, ' Autoencoder')
    predictTime.toc()
    print('Predict time for model '+str(i_train)+ ': ',predictTime.elapsed)
predictTimeT.toc()
print('Total predict time: ', predictTimeT.elapsed)

#%% Multi Plotting
fig = plt.figure(figsize=(8, 6), dpi=80)
markers = ['^', 'x', 'o', 's', 'v', '*']

plt.plot(pOptions,avgGlobalError, color='k', linewidth=lineWidth, linestyle='--', label='No Decoding')
plt.plot(pOptions,avgGlobalErrorMAP, color='k', linewidth=lineWidth, label='MAP')
plt.grid(True, which='both')

for i in range(len(p_trainOptions)):
    plt.scatter(pOptions,multiPredictions[i], marker=markers[i], zorder=3+i, s=markerSize, label='A. Autoencoder, $p_t$ = %s' % p_trainOptions[i])
    
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()

figPath = directory+'\\images'
fn.createDir(figPath)
fig.set_size_inches(width, height)
fig.savefig(figPath+'\MAP_'+title+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)


