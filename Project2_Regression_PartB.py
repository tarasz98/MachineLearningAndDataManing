'''
INTRODUCTION TO MACHINE LEARNING AND DATA MINING - Report 2
'''

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
X_new=X_new.drop('target',axis=1)



#---------Getting dummies for the categorical features--------
X = pd.get_dummies(X_new,columns=['cp','restecg','slope','ca','thal'],drop_first=False)

y = X['thalach']
X = (X - X.mean())/X.std()

attributeNames = np.asarray(X.columns)


X = X.drop('thalach', axis = 1)

X = np.concatenate((np.ones((X.shape[0],1)),X),1)

attributeNames = [u'Offset']+attributeNames
N, M = np.shape(X)




X = np.asarray(X)
y = np.asarray(y)
# REGRESSION


'''
# Values of lambda
lambdas = np.power(10.,range(-5,9))

K = 10
# Simple holdout-set crossvalidation
test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K)
'''

#- Part B: Comparing 3 models

# LINEAR REGRESSION

# Cross Validation Parameters for inner and outer fold.
K_Outer = 5
CV_Outer = model_selection.KFold(K_Outer, shuffle=True)

K_Inner = 5
CV_Inner = model_selection.KFold(K_Inner, shuffle=True)

#Neural Network parameters
h = [15, 20, 25, 30, 35]
max_iter = 10000



# Initialize variables
lambdas = np.power(10., range(-5,9))
optim_lambdas = np.empty(K_Outer)
#T = len(lambdas)
Error_train = np.empty((K_Outer,1))
Error_test = np.empty((K_Outer,1))
Error_train_rlr = np.empty((K_Outer,1))
Error_test_rlr = np.empty((K_Outer,1))
Error_train_nf_in = np.empty((K_Inner,1))
Error_test_nf_in = np.empty((K_Inner,1))
Error_train_nf_out = np.empty((K_Outer,1))
Error_test_nf_out = np.empty((K_Outer,1))
w_rlr = np.empty((M,K_Outer))
mu = np.empty((K_Outer, M-1))
sigma = np.empty((K_Outer, M-1))
w_noreg = np.empty((M,K_Outer))
InnerErrors_h = np.empty((K_Inner, 1))
Min_Error_h = np.empty((K_Inner, 1))
Best_h = h
Error_test_ANN = np.empty((K_Outer, 1))



k_out=0
for train_index, test_index in CV_Outer.split(X,y):
    
    # extract training and test set for current CV fold
    X_train_out = X[train_index]
    y_train_out = y[train_index]
    X_test_out = X[test_index]
    y_test_out = y[test_index]
    internal_cross_validation = 5    
     
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_out, y_train_out, lambdas, internal_cross_validation)
    
    optim_lambdas[k_out] = opt_lambda
    
    Xty = X_train_out.T @ y_train_out
    XtX = X_train_out.T @ X_train_out
    
    
    
    ## REGULARIZED LINEAR REGRESSION
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k_out] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k_out] = (np.square(y_train_out-X_train_out @ w_rlr[:,k_out]).sum(axis=0))/y_train_out.shape[0]
    Error_test_rlr[k_out] = (np.square(y_test_out-X_test_out @ w_rlr[:,k_out]).sum(axis=0))/y_test_out.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k_out] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k_out] = (np.square(y_train_out-X_train_out @ w_noreg[:,k_out]).sum(axis=0))/y_train_out.shape[0]
    Error_test[k_out] = (np.square(y_test_out-X_test_out @ w_noreg[:,k_out]).sum(axis=0))/y_test_out.shape[0]
    
    #X_test_out @ w_noreg[:,k_out] : Y predict
    
    
    # To inspect the used indices, use these print statements
    print('Outer cross validation fold {0}/{1}:'.format(k_out+1,K_Outer))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}\n'.format(test_index))
    
    ## ARTIFICIAL NETWORK
    
    
        
    for k_in, (train_index2, test_index2) in enumerate(CV_Inner.split(X_train_out,y_train_out)): 
        h = [15, 20, 25, 30, 35]    
        # Extract training and test set for current CV fold, 
        # and convert them to PyTorch tensors
        X_train_inner = (X_train_out[train_index2,:] )
        y_train_inner = (y_train_out[train_index2] )
        X_test_inner = (X_train_out[test_index2,:] )
        y_test_inner = (y_train_out[test_index2] ) 
        
        ## BASELINE MODEL ERROR
        
        # Compute mean squared error without using the input data at all
        Error_train_nf_in[k_in] = (np.square(y_train_inner-y_train_inner.mean()).sum(axis=0))/y_train_inner.shape[0]
        Error_test_nf_in[k_in] = (np.square(y_test_inner-y_test_inner.mean()).sum(axis=0))/y_test_inner.shape[0]
                                            #Test         #Predict
                                            
                                            
                                            
        Error_train_nf_out[k_out] = min(Error_train_nf_in)
        Error_test_nf_out[k_out] = min(Error_test_nf_in)
            
        X_train_inner = torch.Tensor(X_train_out[train_index2,:] )
        y_train_inner = torch.Tensor(y_train_out[train_index2] )
        X_test_inner = torch.Tensor(X_train_out[test_index2,:] )
        y_test_inner = torch.Tensor(y_train_out[test_index2] )
            
        for i, j in enumerate(h):  
            
            # Create a model for each h
            ann_model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, h[i]), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            
                            torch.nn.Linear(h[i], 1), # H hidden units to 1 output neuron
                            )
            loss_fn = torch.nn.MSELoss()
            print('\nTesting h: {0}'.format(j))    
        
            # Train the new model
            net, final_loss_in, learning_curve = train_neural_net(ann_model,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
        
            print('\n\tBest loss: {}\n'.format(final_loss_in))
            
            # Determine estimated class labels for test set
            y_pred = net(X_test_inner) # activation of final note, i.e. prediction of network
            y_pred_np = y_pred.detach().numpy()
            y_test_np = y_test_inner.detach().numpy()
            #y_test = y_test.type(dtype=torch.uint8)
            # Determine errors and error rate
            InnerErrors_h[i] = final_loss_in/y_test_inner.shape[0]
            if (InnerErrors_h[i] < InnerErrors_h[i-1]):
                Besth = j
        
        print("Inner cross validation fold: {0}/{1}".format(k_in+1, K_Inner))
        
        #Choose the minimum error for given h
        Min_Error_h[k_in] = min(InnerErrors_h)
        
        # Best h for each inner fold        
        Best_h[k_out] = Besth
    
    # Retrain the ANN with the best h and the outer X_train and y_train
     
    outer_ann = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, np.asarray(Best_h[k_out])), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            
                            torch.nn.Linear(Best_h[k_out], 1), # H hidden units to 1 output neuron
                            )
    loss_fn = torch.nn.MSELoss()
    
    # Training data to pytorch
    X_train_out = torch.Tensor(X[train_index,:] )
    y_train_out = torch.Tensor(y[train_index] )
    X_test_out = torch.Tensor(X[test_index,:] )
    y_test_out = torch.Tensor(y[test_index] )
    
    # Train the net with outer data folds
    net, final_loss_out, learning_curve = train_neural_net(outer_ann,
                                                               loss_fn,
                                                               X=X_train_out,
                                                               y=y_train_out,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
    
    # Error of the outer neural network
    y_test = y_test_out.detach().numpy()  
    y_pred_out = net(X_test_out).detach().numpy()    
    Error_test_ANN[k_out] = final_loss_out/y_test_out.shape[0]
    
    #y_pred_ANN for evaluation of performance
    y_pred_ANN = np.concatenate(y_pred_out)
    
    #y_pred_BL for evaluation of performance
    y_pred_BL = np.array(y_test_out.mean())
    
    p=range(len(y_test_out))
    y_pred_BL=array.array('f',[])
    for i in p:
        y_pred_BL.append(y_test_out.mean().numpy())
    len(y_pred_BL)
    
    #y_pred_RLR for evaluation of performance 
    y_pred_RLR = X_test_out @ w_rlr[:,k_out]
    
    
    k_out+=1
    

'''    
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

'''

#help(train_neural_net)
#for i, j in enumerate(h):
   # print(i, j)
