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
batchSize = trainSize 
timestr = time.strftime("%Y%m%d-%H%M%S")
Mep_trainOptions = np.array([2**17, 2**18])# 2**13, 2**14, 2**15, 2**16])
train_p = 0.07

myDir = 'Simulation'+'_bs_'+str(batchSize)+'_Mepmin_'+str(min(Mep_trainOptions))+'_Mepmax_'+str(max(Mep_trainOptions))
fn.createDir('Decoder_Simulations')
directory = 'Decoder_Simulations\\'+ myDir
fn.createDir(directory)
fn.createDir(directory+'\\ArrayModels')
fn.createDir(directory+'\\1HModels')

paths1 = ["" for x in range(len(Mep_trainOptions))]
paths2 = ["" for x in range(len(Mep_trainOptions))]
trainTimeT.tic()
for i in range(len(Mep_trainOptions)):
    trainTime.tic()
    numEpochs = Mep_trainOptions[i]
        
    # Array decoder
    title = 'MLNN'
    paths1[i] = directory+'\\ArrayModels\\'+'i_'+str(i)+'_'+title+'_Mep_'+str(numEpochs)+'.h5'
    path = paths1[i]
    exec(open("MLNN-train.py").read())
    
    # One hot decoder
    title = 'MLNN1H'
    paths2[i] = directory+'\\1HModels\\'+'i_'+str(i)+'_'+title+'_Mep_'+str(numEpochs)+'.h5'
    path = paths2[i]
    exec(open("MLNN1H-train.py").read())
    
    trainTime.toc()
    print('Train time for models of wave '+str(i)+': ',trainTime.elapsed)
trainTimeT.toc()
print('Total Train time for models: ',trainTimeT.elapsed)
    
    
#%%
'''
    Multi Prediction Array Decoder
'''
predictTime = TicToc('Predict')
predictTimeT = TicToc('PredictT')
# Initialize variables
multiPredictionsA = np.empty([len(Mep_trainOptions),len(pOptions)])
globalReps = 1000
globalErrorADecoder = np.empty([globalReps, len(pOptions)])
predictTimeT.tic()
for i_train in range(len(Mep_trainOptions)):
    predictTime.tic()
    # load weights into new model
    fileName = paths1[i_train]
    loadedModel = tf.keras.models.load_model(fileName)
    print("Loaded model from disk")
    plot_model(loadedModel, to_file='Debug\Loaded.pdf')
    ArrayDecoder = loadedModel.layers[1]
    
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = fn.generateU(N,k)
            x = fn.generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            yflat = fn.BSC(xflat,p)
            y = yflat.reshape(N,2*k) # noisy codewords
            prediction = ArrayDecoder.predict(y)
            predictedMessages = np.round(prediction)
        
            globalErrorADecoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)
            
    multiPredictionsA[i_train] = np.average(globalErrorADecoder,0)
    #% individual Plotting
    title = 'MLNN'
    plotBERp(globalErrorADecoder, 'Array Decoder')
    predictTime.toc()
    print('Predict time for Array Model '+str(i_train)+ ': ',predictTime.elapsed)
predictTimeT.toc()
print('Total predict time: ', predictTimeT.elapsed)

#%% Multi Plotting Array Decoder
fig = plt.figure(figsize=(8, 6), dpi=80)
markers = ['^', 'x', 'o', 's', 'v', '*']

plt.plot(pOptions,avgGlobalError, color='b', linewidth=lineWidth, linestyle='--', label='No Decoding')
plt.plot(pOptions,avgGlobalErrorMAP, color='r', linewidth=lineWidth, label='MAP')
plt.grid(True, which='both')

for i in range(len(Mep_trainOptions)):
    plt.scatter(pOptions,multiPredictionsA[i], marker=markers[i], zorder=3+i, s=markerSize, label='A. Decoder, Mep = %s' % Mep_trainOptions[i])
    
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()

figPath = directory+'\\images'
fn.createDir(figPath)
fig.set_size_inches(width, height)
title = 'MLNN'
fig.savefig(figPath+'\MAP_'+title+'_p_'+str(train_p)+'.png', bbox_inches='tight', dpi=300)

#%%%
'''
    Multi Prediction 1H Decoder
'''
predictTime = TicToc('Predict')
predictTimeT = TicToc('PredictT')
# Initialize variables
multiPredictionsH = np.empty([len(Mep_trainOptions),len(pOptions)])
globalReps = 1000
globalErrorHDecoder = np.empty([globalReps, len(pOptions)])
predictTimeT.tic()
for i_train in range(len(Mep_trainOptions)):
    predictTime.tic()
    # load weights into new model
    fileName = paths2[i_train]
    loadedModel = tf.keras.models.load_model(fileName)
    print("Loaded model from disk")
    plot_model(loadedModel, to_file='Debug\Loaded.pdf')
    HDecoder = loadedModel
    
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = fn.generateU(N,k)
            x = fn.generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            yflat = fn.BSC(xflat,p)
            y = yflat.reshape(N,2*k) # noisy codewords
            prediction = HDecoder.predict(y)
            predictedMessages = fn.multipleOneshot2messages(prediction, messages)
        
            globalErrorHDecoder[i_global][i_p] = fn.bitErrorFunction(predictedMessages, u)
            
    multiPredictionsH[i_train] = np.average(globalErrorHDecoder,0)
    #% individual Plotting
    title = 'MLNN1H'
    plotBERp(globalErrorHDecoder, 'One-hot Decoder')
    predictTime.toc()
    print('Predict time for Array Model '+str(i_train)+ ': ',predictTime.elapsed)
predictTimeT.toc()
print('Total predict time: ', predictTimeT.elapsed)

#%% Multi Plotting 1H Decoder
fig = plt.figure(figsize=(8, 6), dpi=80)
markers = ['^', 'x', 'o', 's', 'v', '*']

plt.plot(pOptions,avgGlobalError, color='b', linewidth=lineWidth, linestyle='--', label='No Decoding')
plt.plot(pOptions,avgGlobalErrorMAP, color='r', linewidth=lineWidth, label='MAP')
plt.grid(True, which='both')

for i in range(len(Mep_trainOptions)):
    plt.scatter(pOptions,multiPredictionsH[i], marker=markers[i], zorder=3+i, s=markerSize, label='1H. Decoder, Mep = %s' % Mep_trainOptions[i])
    
plt.xlabel('$p$')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()

figPath = directory+'\\images'
fn.createDir(figPath)
fig.set_size_inches(width, height)
title = 'MLNN1H'
fig.savefig(figPath+'\MAP_'+title+'_p_'+str(train_p)+'.png', bbox_inches='tight', dpi=300)
