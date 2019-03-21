# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:33:43 2019

@author: Eduardo D
"""
# Import libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt

a = np.linspace(1,100,100)
y = np.sin(a)
plt.plot(a,y)
plt.figure()
plt.plot(y,a)

a.dot(y) # equivalent to dot product

b = np.array([1, 2, 3, 4])
a3 = np.array(b, b ,b)
print(a3)


for i in range(10):
    print(i*rd.randint(0,1))
    