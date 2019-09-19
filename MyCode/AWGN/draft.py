# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:11:47 2019

@author: user
"""
x = K.ones(shape=(100,256))
fn.func_output_shape(x)
y = fn.tensorAWGN(x)
print(K.eval(y))