#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:24:10 2020

@author: n_rishabh
"""

import pandas as pd
import numpy as np

df = pd.read_csv('AI-DataTrain.csv', index_col=0)
X = df
Y = np.sum(df, axis = 0)
Y = 1000-Y
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(-1,1)
Y = Y.T
Y = Y/1000

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def layer_sizes(X, Y):
    n_x = X.shape[0] 
    n_h = 20
    n_y = Y.shape[0] 
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2) 
    
    W1 = np.random.randn(n_h, n_x)*0.01
    W2 = np.random.randn(n_y, n_h)*0.01
    
    assert (W1.shape == (n_h, n_x))
    assert (W2.shape == (n_y, n_h))
    
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def forward_propagation(X, parameters):
   
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    Z1 = np.dot(W1,X)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1)
    A2 = sigmoid(Z2)
    
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    
    
    m = Y.shape[1] # number of example

    
    cost = np.mean(np.square(Y-A2))
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1 - np.power(A1, 2)))
    dW1 = np.dot(dZ1,X.T)/m
    
    grads = {"dW1": dW1,
             "dW2": dW2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.3):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    dW1 = grads['dW1']
    dW2 = grads['dW2']
    
   
    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
   
    parameters = initialize_parameters(n_x, n_h, n_y);
    
    for i in range(0, num_iterations):
         
        
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads, learning_rate = 0.3)
        
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2
    
    return predictions

parameters = nn_model(X, Y, n_h = 20, num_iterations = 2000, print_cost=True)

predictions = predict(parameters, X)
predictions


np.mean(Y)
SS_model = np.sum(np.square(Y-predictions))
SS_model
SS_total = np.sum(np.square(Y-np.mean(Y)))
SS_total
R2 = 1 - SS_model/SS_total
R2
rss=((Y-predictions)**2).sum()
mse=np.mean((Y-predictions)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y-predictions)**2)))

import xlsxwriter

workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()

ques = workbook.add_format({'num_format': 'Q##'})

row = 1
col = 0
n=0

for i in range(50):
    predicts = predictions[0][n]
    worksheet.write(row, col, row, ques)
    worksheet.write(row, col+1, predicts)
    row += 1
    n += 1

workbook.close()

df = pd.read_csv('AI-DataTest.csv')
X = df
Y = np.sum(df, axis = 0)
Y = 50-Y
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(-1,1)
Y = Y.T
Y=Y/50
parameters = nn_model(X, Y, n_h = 10, num_iterations = 2000, print_cost=True)

pred = predict(parameters, X)
pred

np.mean(Y)
SS_model = np.sum(np.square(Y-pred))
SS_model
SS_total = np.sum(np.square(Y-np.mean(Y)))
SS_total
R2 = 1 - SS_model/SS_total
R2
rss=((Y-pred)**2).sum()
mse=np.mean((Y-pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y-pred)**2)))

import xlsxwriter

workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

ques = workbook.add_format({'num_format': 'Q##'})

row = 0
col = 0
n=0
q = 10

for i in range(25):
    predicts = pred[0][n]
    worksheet.write(row, col, q, ques)
    worksheet.write(row, col+1, predicts)
    row += 1
    n += 1
    q +=1

workbook.close()



