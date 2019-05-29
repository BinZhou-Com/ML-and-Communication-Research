#%%
'''
       Load libraries
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import erfc  # complementary error function

import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K
from keras.utils import plot_model

import sys
sys.path.append('C:\\Users\\user\\Desktop\\GitHub\\PIR\\MyCode')
import my_functions as fn

import time

#Plot setup
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.weight':'normal'})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)

# width as measured in inkscape
width = 3.487
height = width / 1.618


#import matplotlib
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()

'''
       Load constants
'''
N = 100 # number of messages sent

# Message characteristics
## Hamming coding
m = 4 # parity bits
n = 2**m-1 # total bits of one codeword
k = n-m # data bits
n = 16
k = 8
d = n-k+1
R = k/n # rate
name = np.array([n, k]) # Hamming(n,k)
#H = fn.parityCheckMatrix(name) # systematic form
#G = fn.matrixGenerator(H,name)

Gdef = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
G = Gdef
#print("Parity check: \n",dot(H.T,G)%2)

messages, possibleCodewords = fn.possibleCodewordsG(name, G)

# Channel
CTYPE = "BSC"
pOptions = np.arange(0.005, 0.105, 0.005) # cross probability of BSC
He = fn.Hb(pOptions)# binary entropy function
C = 1-He # channel capacity (R<C)

# Simulation
globalReps = 1000

#% Global Functions
def tensorBSC(x):
    # GLOBAL value of p: optimal training statistics for neural based channel decoders (paper)
    p = K.constant(train_p,dtype=tf.float32)
    var = K.random_uniform(shape=(fn.func_output_shape(x),), minval = 0.0, maxval=1.0)
    noise = K.less(var, p)
    noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noiseFloat, x)%2
    return result

def plotBERp(globalErrorMLNN, legendEntry):
    avgGlobalErrorMLNN = np.average(globalErrorMLNN,0)
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.scatter(pOptions,avgGlobalErrorMLNN, color='g', marker='^', zorder=3, s=8)
    plt.plot(pOptions,avgGlobalError, color='b', linewidth=1, linestyle='--')
    plt.plot(pOptions,avgGlobalErrorMAP, color='r', linewidth=1)
    plt.grid(True, which='both')
    #plt.title('Batch size = '+str(batchSize)+', train_p = ' + str(train_p))
    plt.xlabel('$p$')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.legend(['No Decoding', 'MAP', legendEntry+ ', $M_{ep}=$'+str(numEpochs)])
    plt.show()
    
    fig.set_size_inches(width, height)
    fig.savefig('images/'+title+'/'+timestr+'_MAP_'+title+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)

