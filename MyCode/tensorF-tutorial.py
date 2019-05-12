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

#%%
''' 
       Training model
'''

MLNN.compile(optimizer=tf.train.AdamOptimizer(0.001),
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])

#%%
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

#%% Import datasets using Datasets API
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

#%% Evaluate and predict
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












       


