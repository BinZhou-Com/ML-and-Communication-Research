# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:12:53 2019

@author: user
"""
'''
    Decoding
'''
# Syndrome decoding
S = fn.syndrome(y, H, name)

D = y # error corrected codewords
e = zeros([N,n])# bit error location
# single error correction
for i in range(N):
    if(sum(S[i,:])!=0):
        #print(i)
        index = sum(S[i,:] == H.T,1).argmax() # Find position where H matrix says there is an error
        e[i,int(index)] += 1
e = e%2
D = (D+e)%2 # decoded codewords

E2cw, globalErrorHamming[i_global][i_p] = fn.errorFunction(D, x, u, name)


'''
    Show quickly if the algo is working
'''

print("# of binary errors without correction: \n", globalError[i_global][i_p])
#print("# of binary errors with Syndrome correction: \n", globalErrorHamming[i_global][i_p])
print("# of binary errors with MAP correction: \n", globalErrorMAP[i_global][i_p])

