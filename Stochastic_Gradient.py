#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:07:12 2017

@author: benjelloun
"""
import numpy as np
import numpy.random as npr
from pylab import scatter, plt

####################################################################################
############################ Generate Data : 
###### Generate Data_X : uniformly drawn in [0, 1]
## Uniformly drawn in [0, 1]
## X_i in  [0, 1]^2
###### Generer Data_Y : 
## y_i = sign(<w*, x>) in {-1, 1} wwhere w* is chosen 
####################################################################################
def generate_data_uniform(N = 1000, M = 2):

    return npr.rand( M, N) #M*N matrix X_i = Data_X[:, i]
    
def generate_labels( Data_X, W = [1, -1]):
    
    
    return [ 2*int( np.dot(W, Data_X[:,i]) > 0) - 1 for i in range(Data_X.shape[1])] 

####################################################################################
## plot_data_labels funciton that takes data : X and labels : Y
## gives plot of this data labeled 
####################################################################################                  
def plot_data_labels(Data, labels):
    
    colors = ['r', 'b']
    scatter(Data[0, :], Data[1, :], color = [colors[int(y > 0)] for y in labels])
    
    return


############################ Perform Stochastic Gradient ############################
#### Classification function : < W, X >
#### The empirical loss :  Rn(w) = (1/n)*SUM_{i=1:n} ( y_i − transpose(w)*x_i )^2
#### stochastic gradient to minimize Rn
####################################################################################
def gradient_Stochastic( Data_X, Data_Y, K = 10):
    
    w = Data_X[:, 0]  # Initialization

    for k in np.arange(K):
        
        i = int(Data_X.shape[1]*npr.rand())  #Choix aléatoire de i
        w = w - (2./(k+1))*(  - Data_Y[i]*Data_X[:,i] +  np.dot(Data_X[:,i], w)*Data_X[:,i]  )
        
    return w

    
##############################  Main :
#Set the numbers N, M, K
N = 1000
M = 2
K = 100
# generate the Data Data_X and True_Y with a chosen w
Data_X = generate_data_uniform(N, M)
W = [1, -1]

############### Firs case : Y = < W, X > ################
True_Y = generate_labels(Data_X, W)
#plot the map (X, Y)
plot_data_labels(Data_X, True_Y)
## Call stochastic gradient to get the learnng function <w, x> :
linear_prediction_w = gradient_Stochastic( Data_X, True_Y, K)
linear_prediction_label = generate_labels( Data_X, linear_prediction_w)
plot_data_labels(Data_X, linear_prediction_label)

############### Second case : Y = < W, X > ################
Y = [ 2*int( np.dot(W, Data_X[:,i]) + npr.randn()/5 > 0) - 1 for i in range(Data_X.shape[1])] 
#plot the map (X, Y)
plot_data_labels(Data_X, Y)
## Call stochastic gradient to get the learnng function <w, x> :
linear_prediction_w = gradient_Stochastic( Data_X, Y, K)
linear_prediction_noise = generate_labels( Data_X, linear_prediction_w)
plot_data_labels(Data_X, linear_prediction_noise)

 


