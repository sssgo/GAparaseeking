# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:44:29 2017

@author: LSJ
"""
import numpy as np 

def  ActivationFunc( tempH, ActivationFunction,p):
    def relu(x,p):
        x[x<0]=0
        return x
    switch ={
        'sig' : lambda x,p:1/(1+np.exp(-p*x)),
        'sin' : lambda x,p:np.sin(x),
        'relu' : relu,
        'srelu' : lambda x,p : np.log(1+np.exp(x)),
        'tan' : lambda x,p : np.tanh(p*x),
    }
    fun = switch.get(ActivationFunction)
    return fun(tempH,p)

    