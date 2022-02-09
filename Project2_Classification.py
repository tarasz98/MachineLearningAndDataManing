# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:11:45 2021

@author: Usuario
"""

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import numpy as np
import pandas as pd
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
from scipy import stats
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import torch
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statistics as st
import array 

# Dataset declaration and data cleasing

df=pd.read_csv('heart.csv')
df.shape
df.info()
df.isnull().sum()
df.info()

# Creation of matrix X

X = df.values

# Outlier removal

z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = df[filtered_entries]

X_new=new_df
y = X_new['target']
X_new=X_new.drop('target',axis=1)

#---------Getting dummies for the categorical features--------
X = pd.get_dummies(X_new,columns=['cp','restecg','slope','ca','thal'],drop_first=False)


X = (X - X.mean())/X.std()

attributeNames = np.asarray(X.columns)

N, M = np.shape(X)

X = np.asarray(X)
y = np.asarray(y)

#CLASSIFICATION-------------------------

# Cross Validation Parameters for inner and outer fold.
K_Outer = 5
CV_Outer = model_selection.KFold(K_Outer, shuffle=True)

K_Inner = 5
CV_Inner = model_selection.KFold(K_Inner, shuffle=True)

#Neural Network parameters
h = [1, 3, 5, 7, 9]
max_iter = 10000

#--MODEL ERRORS
Error_test_LR=np.empty((K_Inner,1))
opt_lambda_idx=np.empty((K_Inner,1))
opt_lambda=np.empty((K_Inner,1))

Error_train_bl_in = np.empty((K_Inner, 1))
Error_test_bl_in = np.empty((K_Inner, 1))
Error_train_bl_out = np.empty((K_Outer, 1))
Error_test_bl_out = np.empty((K_Outer, 1))

Error_ANN_h = np.empty((K_Inner, 1))
error_in = []
error_out = []
Best_h = np.empty((K_Outer, 1))
Min_Error_h = np.empty((K_Inner, 1 ))
Error_ANN_out = []



## ----OUTER CROSS VALIDATION FOLD

k_out=0
for train_index, test_index in CV_Outer.split(X,y):
    print('Outer cross validation fold {0}/{1}:'.format(k_out+1,K_Outer))
    
    # Extract training and test set for the outer cross validation fold
    X_train_outer = X[train_index]
    y_train_outer = y[train_index]
    X_test_outer = X[test_index]
    y_test_outer = y[test_index]

    # Fit regularized logistic regression model to training data to predict 

    lambda_interval = np.logspace(-8, 2, 50)
    optim_lambdas = np.empty(K_Outer)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    
    
    ## -----INNER CROSS VALIDATION FOLD
    
    k_in=0
    for train_index2, test_index2 in CV_Inner.split(X_train_outer,y_train_outer):
        h = [1, 3, 5, 7, 9]
        print('Inner cross validation fold {0}/{1}:'.format(k_in+1,K_Inner))
        
        # Extract inner training and test set for current CV fold
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(X_train_outer, y_train_outer, test_size=.80)
        
        
        #----BASELINE MODEL
        Error_train_bl_in[k_in] = np.sum(y_train_inner != np.argmax(np.bincount(y_train_inner)))/len(y_train_inner)
        Error_test_bl_in[k_in] = np.sum(y_test_inner != np.argmax(np.bincount(y_test_inner)))/len(y_test_inner)
        
        
        
        #vector = np.vectorize(np.int)
        #vector(y_test_inner.numpy())  
        
        
        #----LOGISTIC REGRESSION CLASSIFICATION
        
        # Selection of the best lambda for the inner cross validation fold
        for k in range(0, len(lambda_interval)):
            
            #Creation of the Logistic Regression Model
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
        
            # Training of the model with the inner partition of the CV
            mdl.fit(X_train_inner, y_train_inner)
    
            # Prediction of the model on the inner test partitions
            y_train_est = mdl.predict(X_train_inner).T
            y_test_est = mdl.predict(X_test_inner).T #y_predict
            
            
            # Compute the model erro for each lambda
            train_error_rate[k] = np.sum(y_train_est != y_train_inner) / len(y_train_inner)
            test_error_rate[k] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)
    
            w_est = mdl.coef_[0] 
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
        
        #----ARTIFICIAL NEURAL NETWORK FOR CLASSIFICATION
        X_train_inner = torch.Tensor(X_train_outer[train_index2,:] )
        y_train_inner = torch.Tensor(y_train_outer[train_index2] )
        X_test_inner = torch.Tensor(X_train_outer[test_index2,:] )
        y_test_inner = torch.Tensor(y_train_outer[test_index2] )
        
        y_train_inner = y_train_inner.unsqueeze(1)
        error_in = []
            
        for i, j in enumerate(h):  
            
            # Create a model for each h
            inner_ann = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, h[i]), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            
                            torch.nn.Linear(h[i], 1), # H hidden units to 1 output neuron
                            torch.nn.Sigmoid() #Final transfer function
                            )
            loss_fn = torch.nn.BCELoss()
            print('\nTesting h: {0}'.format(j))    
            
            
            # Train the new model
            net, final_loss_in, learning_curve = train_neural_net(inner_ann,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
        
            print('\n\tBest loss: {}\n'.format(final_loss_in))
            
            # Determine estimated class labels for test set
            y_sigmoid_in = net(X_test_inner) # activation of final note, i.e. prediction of network
            y_test_est_in = (y_sigmoid_in > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
            y_test_in = y_test_inner.type(dtype=torch.uint8)
            # Determine errors and error rate
            e_in = (y_test_est_in != y_test_in)
            error_rate_in = (sum(e_in).type(torch.float)/len(y_test_inner)).data.numpy()
            error_in.append(error_rate_in) # store error rate for current CV fold
            Error_ANN_h[i] = round(np.mean(error_in),4)
            # Determine errors and error rate
            #InnerErrors_h[i] = final_loss_in/y_test_inner.shape[0]
            if (Error_ANN_h[i] < Error_ANN_h[i-1]):
                Besth = j
            else:
                Besth = h[0]
            
        
        
        #Choose the minimum error for given h
        Min_Error_h[k_in] = min(Error_ANN_h)
        
        # Best h for each inner fold        
        Best_h[k_out] = Besth
        
        
        k_in+=1
    
    # COMPUTE THE ERRORS OF THE BEST MODEL FOR THE OUTER FOLD
    
    # Baseline Model
    Error_train_bl_out[k_out] = min(Error_train_bl_in)
    Error_test_bl_out[k_out] = min(Error_test_bl_in)
    
    p=range(len(y_test_outer))
    y_predict_bl=array.array('i',[])
    for i in p:
        y_predict_bl.append(np.argmax(np.bincount(y_test_outer)))
    len(y_predict_bl)
    
    # Logistic Regression
    Error_test_LR[k_out] = np.min(test_error_rate)
    opt_lambda_idx[k_out] = np.argmin(test_error_rate)
    opt_lambda[k_out] = lambda_interval[int(opt_lambda_idx[k_out])]
    
    
    
    LR= LogisticRegression(penalty='l2' , C=1/opt_lambda[k_out].item() )

    LR.fit(X_train_outer, y_train_outer)
    
    y_predict_LR = LR.predict(X_test_outer).T
    

    
    
    # Neural Network for outer fold
    # - Create Outer ANN model
    outer_ann = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, int(np.asarray(Best_h[k_out]))), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            
                            torch.nn.Linear(int(np.asarray(Best_h[k_out])), 1), # H hidden units to 1 output neuron
                            torch.nn.Sigmoid() #Final transfer function
                            )
    loss_fn = torch.nn.BCELoss()
    
    # - Training data to pytorch
    X_train_out = torch.Tensor(X[train_index,:] )
    y_train_out = torch.Tensor(y[train_index] )
    X_test_out = torch.Tensor(X[test_index,:] )
    y_test_out = torch.Tensor(y[test_index] )
    
    
    # - Train the net with outer data folds
    y_train_out = y_train_out.unsqueeze(1)
    net, final_loss_out, learning_curve = train_neural_net(outer_ann,
                                                               loss_fn,
                                                               X=X_train_out,
                                                               y=y_train_out,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
    
    # - Compute the errors of the ANN
    # -- Determine estimated class labels for test set
    y_sigmoid_out = net(X_test_out) # activation of final note, i.e. prediction of network
    
    y_test_est_out = (y_sigmoid_out > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function

    y_predict_ANN = np.concatenate(y_test_est_out.numpy())
    
    y_test_out = y_test_out.type(dtype=torch.uint8)
    
    # -- Determine errors and error rate
    e_out = (y_test_est_out != y_test_out)
    error_rate_out = (sum(e_out).type(torch.float)/len(y_test_out)).data.numpy()
    Error_ANN_out.append(error_rate_out) # store error rate for current CV fold
    Error_ANN_out[k_out] = round(np.mean(error_in),4)
    
    
    k_out+=1
















