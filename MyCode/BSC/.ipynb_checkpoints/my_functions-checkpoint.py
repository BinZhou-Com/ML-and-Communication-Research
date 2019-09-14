# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:28:03 2019

@author: Eduardo D
"""

from numpy import *
import matplotlib.pyplot as plt

from scipy.special import erfc # complementary error function

def plotit(t, u, title="Title", X="X", Y="Y"):
    plt.figure()
    plt.plot(t,u,'b')
    
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()

    
