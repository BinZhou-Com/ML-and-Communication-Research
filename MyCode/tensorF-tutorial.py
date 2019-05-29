# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:22:42 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import time
import datetime
import os

print(tf.VERSION)
print(tf.keras.__version__)

#%% Tensor board
'''
       Tensor Board
'''
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard\


log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
'''
       Sequential Model: most simple tf MLNN model
'''
MLNN = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with 64 units to the model:
              layers.Dense(64, activation='relu', input_shape=(32,)),
              # Add another:
              layers.Dense(64, activation='relu'),
              # Add a softmax layer with 10 output units:
              layers.Dense(10, activation='softmax')
])


''' 
       Training model
'''

MLNN.compile(optimizer=tf.train.AdamOptimizer(0.001),
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])


'''
    Import data from numpy and TRAIN the model
'''
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

#hitory = MLNN.fit(data, labels, epochs=10, batch_size=32) #  fits to the training data

MLNN.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels), callbacks=[tensorboard_callback])

plt.plot(history.history['loss'])


'''
    Import datasets using Datasets API
'''

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
MLNN.fit(dataset, epochs=10, steps_per_epoch=30)

MLNN.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)


'''
        evaluate the inference-mode 
''' 

MLNN.evaluate(data, labels, batch_size=32)
MLNN.evaluate(dataset, steps=30)

'''
       predict the output of the last layer in inference for the data provided
'''

result = MLNN.predict(data, batch_size=32)
print(result.shape)

#%% Udacity example 
'''
       General Plan
'''
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1], activation='relu') 
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
model.predict([100.0]) # expected 212
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#%% Tensorboard setup tutorial
'''
       Tensor flow keynote
'''
# tf.summary.FileWriter (STORE_PATH, sess.graph): # A python class that writes data for TensorBoard

# tf session
with tf.Session() as sess:
    print(sess.run(h))

writer = tf.summary.FileWriter("/log")

# use summaries to generate outputs for tensorboard

#%%
'''
    Book tutorial
'''

# Definitions of the graph (called Tensors)
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a,b, name="mul_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")

# Session that will actually do something
# Session objects are in charge of supervising graphs as they run, and are the 
# primary interface for running graphs
sess = tf.Session() #= tf.Session(graph=tf.get_default_graph())
# Performs the computations needed to initialize Variables, but returns `None`
sess.run(tf.global_variables_initializer())
output1=sess.run(e) # calculates the value of a desired tensor


# Loading tensorboard
# SummaryWriter object
# line of code to run in cmd: tensorboard --logdir="my_graph"
# then go to: localhost:6006
writer = tf.summary.FileWriter('./my_graph', sess.graph)

writer.close()
sess.close()


a = tf.constant([5,3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b,c, name="add_d")

# Data types: numpy is recommended for specifying tensors by hand
# 0-D Tensor with 32-bit integer data type
t_0 = np.array(50, dtype=np.int32)
# 1-D Tensor with byte string data type
# Note: don't explicitly specify dtype when using strings in NumPy
t_1 = np.array([b"apple", b"peach", b"grape"])
# 1-D Tensor with boolean data type
t_2 = np.array([[True, False, False],
                [False, False, True],
                [False, True, False]],
                dtype=np.bool)
# 3-D Tensor with 64-bit integer data type
t_3 = np.array([[ [0, 0], [0, 1], [0, 2] ],
                [ [1, 0], [1, 1], [1, 2] ],
                [ [2, 0], [2, 1], [2, 2] ]],
                dtype=np.int64)
test = np.array([0, 1, 0], dtype=np.int8)

# Tensor shapes
# Shapes that specify a 0-D Tensor (scalar)
# e.g. any single number: 7, 1, 3, 4, etc.
s_0_list = []
s_0_tuple = ()
# Shape that describes a vector of length 3
# e.g. [1, 2, 3]
s_1 = [3]
# Shape that describes a 3-by-2 matrix
# e.g [[1 ,2],
# [3, 4],
# [5, 6]]
s_2 = (3, 2)
# Shape for a vector of any length:
s_1_flex = [None]
# Shape for a matrix that is any amount of rows tall, and 3 columns wide:
s_2_flex = (None, 3)
# Shape of a 3-D Tensor with length 2 in its first dimension, and variable-
# length in its second and third dimensions:
s_3_flex = [2, None, None]
# Shape that could be any Tensor
s_any = None
mystery_tensor = np.array([[1],[3],[[[[[[345]]]]]]])
# Find the shape of the mystery tensor
#shape = tf.shape(mystery_tensor, name="mystery_shape")

# Create a new graph, constructor doesnt have any parameter
g = tf.Graph()

# add operations to a specific graph
with g.as_default():
    # Create Operations as usual; they will be added to graph `g`
    a = tf.multiply(2, 3)
       

# Feed dictionary
    # Create Operations, Tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)
# Start up a `Session` using the default graph
sess = tf.Session()
# Define a dictionary that says to replace the value of `a` with 15
replace_dict = {a: 15}
# Run the session, passing in `replace_dict` as the value to `feed_dict`
sess.run(b, feed_dict=replace_dict) # returns 45

sess.close()

# Other use of session
with tf.Session() as sess:
    # Run graph, write summary statistics, etc.
    sess.run(b)
# The Session closes automatically
    
# placeholder operation
# Creates a placeholder vector of length any with data type int32
a = tf.placeholder(tf.int32, shape=[None], name="my_input")
# Use the placeholder as if it were any other Tensor object
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
# Finish off the graph
d = tf.add(b, c, name="add_d")

# Open a TensorFlow Session
sess = tf.Session()
# Create a dictionary to pass into `feed_dict`
# Key: `a`, the handle to the placeholder's output Tensor
# Value: A vector with value [5, 3] and int32 data type
input_dict = {a: np.array([5, 3], dtype=np.int32)}
# Fetch the value of `d`, feeding the values of `input_vector` into `a`
sess.run(d, feed_dict=input_dict)


# Variables and useful vectors
# Pass in a starting value of three for the variable
my_var = tf.Variable(3, name="my_variable")
add = tf.add(5, my_var)
init = tf.global_variables_initializer()
sess.run(init)
my_var.assign(10) # change the value of the variable

# 2x2 matrix of zeros
zeros = tf.zeros([2, 2])
# vector of length 6 of ones
ones = tf.ones([6])
# 3x3x3 Tensor of random uniform values between 0 and 10
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
# 3x3x3 Tensor of normally distributed numbers; mean 0 and standard deviation 2
normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)
# No values below 3.0 or above 7.0 will be returned in this Tensor (cuts 2 stddev from the mean)
trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)
sess.run(trunc)


# Name scope: used to organize your graph
with tf.name_scope("Scope_A"):
    a = tf.add(1, 2, name="A_add")
    b = tf.multiply(a, 3, name="A_mul")
with tf.name_scope("Scope_B"):
    c = tf.add(4, 5, name="B_add")
    d = tf.multiply(c, 6, name="B_mul")
e = tf.add(b, d, name="output")
sess2 = tf.Session()
sess2.run(e)
writer2 = tf.summary.FileWriter('./name_scope_1', graph=tf.get_default_graph())
writer2.close()

#%%
## CREATING THE GRAPH
graph = tf.Graph()
with graph.as_default():
    
    with tf.name_scope("variables"):
            # Variable to keep track of how many times the graph has been run
           global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
            # Variable that keeps track of the sum of all output values over time:
           total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")
    
    with tf.name_scope("Transformation"):
       # Separate input layer
        with tf.name_scope("input"):
           # Create input placeholder- takes in a Vector
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
      
        # Separate middle layer
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")
       # Separate output layer
       
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")
            
    with tf.name_scope("update"):
        # Increments the total_output Variable by the latest output
        update_total = total_output.assign_add(output)
        # Increments the above `global_step` Variable, should be run whenever the graph is run
        increment_step = global_step.assign_add(1)
        
    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
        # Creates summaries for output node
        tf.summary.scalar('Output', output)
        tf.summary.scalar('Sum_of_outputs_over_time', update_total)
        tf.summary.scalar('Average_of_outputs_over_time', avg)
       
    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.initialize_all_variables()
        # Merge all summaries into one Operation
        merged_summaries = tf.summary.merge_all()
    
## RUNNING THE GRAPH    
sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./improved_graph', graph=graph)
# initialize our variables
sess.run(init)

# helper function to run our graph
def run_graph(input_tensor):
    """
    Helper function; runs the graph with given input tensor and saves summaries
    """
    feed_dict = {a: input_tensor}
    # _ ignores the value of output and create variables step and summary
    _, step, summary = sess.run([output, increment_step, merged_summaries], 
    feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)

# running graph with various inputs
run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])
run_graph([4,1])
run_graph([7,3,1])
run_graph([6,3])
run_graph([0,2])
run_graph([4,5,6])

# saving to disk:
writer.flush()
writer.close()
sess.close()

#%%%
'''
    Machine Learning
'''
## BASIC INFERENCE MODEL

# initialize variables/model parameters
# define the training loop operations
def inference(X):
# compute inference model over data X and return the result
def loss(X, Y):
# compute loss over training data X and expected outputs Y
def inputs():
# read/generate input training data X and expected outputs Y
def train(total_loss):
# train / adjust model parameters according to computed total loss
def evaluate(sess, X, Y):
# evaluate the resulting trained model

# Create a saver - Checkpoint file.
saver = tf.train.Saver()
    
# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # actual training loop
    training_steps = 1000
    
    initial_step = 0
    # verify if we don't have a checkpoint saved already
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
    
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print "loss: ", sess.run([total_loss])
        if step % 1000 == 0:
            saver.save(sess, 'my-model', global_step=step)
    
    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    saver.save(sess, 'my-model', global_step=training_steps)
    sess.close()
    
#%%%
''' 
    Tensor flow own class
'''

from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
   
# Non trainable
class ComputeSum(layers.Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=np.random.binomial(1, 0.5, size=input_dim),
                             trainable=False, dtype=float32)

    def call(self, inputs):
        self.total.assign_add(inputs, self.total)
        return self.total

sess = tf.Session()

my_sum = ComputeSum(2)
init = tf.initialize_all_variables()

sess.run(init)
x = tf.ones(2)
y = my_sum(x)
sess.run(y)

#%%
from keras import backend as K
import numpy as np

def tensorBSC(x):
    # value of p: optimal training statistics for neural based channel decoders (paper)
    p = K.constant(0.07,dtype=tf.float32)
    var = K.random_uniform(shape=(func_output_shape(x),), minval = 0.0, maxval=1.0)
    noise = K.less(var, p)
    noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noiseFloat, x)%2
    return result

def func_output_shape(x):
    shape = x.get_shape().as_list()[0]
    return shape
    
inputTensor = tf.Variable(np.random.binomial(1, 0.5, size=16), dtype=np.float32)
outputTensor = tensorBSC(inputTensor);
shapeTest = func_output_shape(inputTensor)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(inputTensor))
#print(sess.run(noise))
print(sess.run(outputTensor))
print(sess.run(inputTensor))
sess.close()

#%%

x = tf.Variable(np.random.binomial(1, 0.5, size=1600), dtype=np.float32)
p = K.constant(0.35,dtype=tf.float32)
var = K.random_uniform(shape=(1600,), minval = 0.0, maxval=1.0)
noise = K.less(var, p)
noiseFloat = K.cast(noise, dtype=tf.float32)
result = tf.math.add(noiseFloat, x)%2

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(x))
print(sess.run(result))
print(sess.run(metricBER(result, x)))


sess.close()

#%%
def metricBER(y_true, y_pred):
    return K.mean(K.not_equal(y_true,y_pred))

def metricBER1H(y_true, y_pred):
    return K.mean(K.not_equal(y_true,K.round(y_pred)))

def tensorOnehot2singleMessage(h):
    #index = K.cast(K.argmax(h), dtype=tf.int32)
    index = K.argmax(h)
    aux = K.constant(index, dtype=tf.int32)
    a_bin = tf.mod(tf.bitwise.right_shift(tf.expand_dims(aux,1), tf.range(8)), 2)
    return a_bin

y_true = K.variable(np.random.binomial(1, 0.5, size=16), dtype=np.float32)
y_pred = K.variable(np.random.binomial(1, 0.5, size=16), dtype=np.float32)
result = metricBER(y_true, y_pred)

y_pred1H = K.variable(K.round(prediction[0]))
y_true1H = K.variable(u[0])
index = tensorOnehot2singleMessage(y_pred1H)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(y_true))
#print(sess.run(y_pred))
#print(sess.run(result))
print('============')
print(sess.run(y_true1H))
print(sess.run(y_pred1H))
print(sess.run(index))

#%%
sess.close()

sess = tf.Session()
with sess.as_default():
    tensor = tf.constant([10,100])
    index = K.argmax(tensor)
    print('rola', index.eval())
    aux = K.constant([2], dtype=tf.int32)
    #a_dec = tf.constant([aux], dtype=tf.int32)
    a_bin = tf.mod(tf.bitwise.right_shift(tf.expand_dims(aux,1), tf.range(8)), 2)
    formatted = tf.strings.format("{}", tensor)
    out0 = sess.run(tensor)
    out1 = sess.run(index)
    out2 = sess.run(a_bin)

print(out0)
print(out1)
print(out2)

#%% Stack overflow question
sess = tf.Session()
with sess.as_default():
    a_dec = K.constant([0,1,2], dtype=tf.int32)
    index = K.constant([K.argmax(a_dec)], dtype=tf.int32)
    a_bin = tf.mod(tf.bitwise.right_shift(index, tf.range(8)), 2)
    out = sess.run(a_bin)
print(out)
    
#%% 
'''
    Encode data to one-hot
'''
data = np.array([[0,0],[0 ,1], [1,0], [1,1]])
dataSingle = np.array([1, 0])
encoded = tf.keras.utils.to_categorical(data)
encodedSingle = tf.keras.utils.to_categorical(dataSingle)
print(data)
#print(encoded)
print(encodedSingle)

decoded = np.argmax(encodedSingle, axis=1)
print(decoded)

#%%
#binary_string = tf.constant([1, 0, 0, 1, 1])
binary_string = np.array([1, 0, 0, 1, 1])

result = tf.reduce_sum(
    tf.cast(tf.reverse(tensor=binary_string, axis=[0]), dtype=tf.int64)
    * 2 ** tf.range(tf.cast(tf.size(binary_string), dtype=tf.int64)))


with tf.Session():
    print(K.round(result).eval())
    
#%%
def testBER(y_true, y_pred):
    index1 = K.argmax(y_true)
    index2 = K.argmax(y_pred)
    return K.mean(K.not_equal(index1,index2))

sess = tf.Session()
with sess.as_default():
    var = Decoder.weights[2]
    sess.run(tf.global_variables_initializer())
    out = sess.run(var)
print(out)

#%%%
from tensorflow.keras.layers import *

u_train_labels = messages.copy()
x_train_data = possibleCodewords.copy()

numEpochs = 2*10
batchSize = 256

input_layer = Input(shape=(16,))
x = Dense(32)(input_layer)
x = Activation('relu')(x)

x = Flatten()(x)

x_1 = Dense(64)(x)
x_1_stop_grad = Lambda(lambda x: K.stop_gradient(x))(x_1)
x_1 = Dense(32)(x_1_stop_grad)
x_1 = Dense(8)(x_1)

model = tf.keras.Model(inputs=input_layer, outputs=x_1)
model.compile(optimizer='adam', loss='mse')
hisotry = model.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize)
plt.plot(history.history['loss'])


