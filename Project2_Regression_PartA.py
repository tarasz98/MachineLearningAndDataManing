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
from sklearn.preprocessing import StandardScaler


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

X = (X - X.mean())/X.std()

attributeNames = np.asarray(X.columns)
M = len(attributeNames) + 1

y = X['thalach']
X = X.drop('thalach', axis = 1)


X = np.asarray(X)
y = np.asarray(y)
# REGRESSION

# - Part A: Regularization and linear regression

# Regularization

## Crossvalidation
# Create crossvalidation partition for evaluation



# Values of lambda
lambdas = np.power(10.,range(-5,9))

K = 10
# Simple holdout-set crossvalidation
test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K)


figure(1, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
legend(attributeNames[1:], loc='best')
        
subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

Xty = X_train.T @ y_train
XtX = X_train.T @ X_train
    
# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M-2)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
Error_train_rlr = np.square(y_train-X_train @ w_rlr).sum(axis=0)/y_train.shape[0]
Error_test_rlr = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]


print('Weights in last fold:')
for m in range(M-2):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m],2)))









