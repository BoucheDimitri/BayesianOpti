import numpy as np
import cho_inv
import prediction_formulae as pred 
import gp_tools
import test_functions as test_func 




#Test for gp_tools
n = 100
xtest = np.random.rand(n, 2)
theta_vec = [5, 5]
p = 1.5
R = gp_tools.kernel_mat(xtest, theta_vec, p)
print(R)
xnew = np.random.rand(2)
rx = gp_tools.kernel_rx(xtest, xnew, theta_vec, p)
print(rx)



#Test for test_func
y = np.zeros((n, 1))
for i in range(0, n):
	y[i, 0] = test_func.mystery_vec(xtest[i, :])
print(y)



#Test for cho_inv
Rinv = cho_inv.cholesky_inv(R)
print(np.dot(Rinv, R))


#Test for prediction_formulae
beta = pred.beta_est_bis(y, Rinv)
print(beta)
y_hat = pred.y_est_bis(rx, y, Rinv, beta)
print(y_hat)


