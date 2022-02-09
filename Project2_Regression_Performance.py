from Project2_Regression_PartB import *
import scipy.stats as st

# perform statistical comparison of the models

alpha = 0.05

# compute z with squared error.
zA = np.abs(y_test_out.numpy() - y_pred_BL ) ** 2   
zB = np.abs(y_test_out.numpy() - y_pred_ANN) ** 2
zC = np.abs(y_test_out.numpy() - y_pred_RLR.numpy()) ** 2

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), len(z)-1)  # p-value
print("Baseline vs Artificial Neural Network")
print("CI: ", CI)
print("p-value: ", p)
print()

z = zA - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), len(z)-1)  # p-value
print("Baseline vs Linear Regression")
print("CI: ", CI)
print("p-value: ", p)
print()

z = zB - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), len(z)-1)  # p-value
print("Artificial Neural Network vs Linear Regression")
print("CI: ", CI)
print("p-value: ", p)


#We are predicting (and so evaluating model performance) for the last outer fold