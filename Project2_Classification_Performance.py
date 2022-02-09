from toolbox_02450 import mcnemar
from Project2_Classification import *

# Compute the Jeffreys interval
alpha = 0.05

print("Artificial Neural Network vs Linear Regression")
[thetahat, CI, p] = mcnemar(y_test_outer, y_predict_ANN, y_predict_LR, alpha=alpha)
print()

print("Artificial Neural Network vs Baseline")
[thetahat, CI, p] = mcnemar(y_test_outer , y_predict_ANN, y_predict_bl, alpha=alpha)
print()

print("Logistic Regression vs Baseline")
[thetahat, CI, p] = mcnemar(y_test_outer, y_predict_LR, y_predict_bl, alpha=alpha)
print()


#We are predicting (and so evaluating model performance) for the last outer fold