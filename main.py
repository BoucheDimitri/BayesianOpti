import numpy as np
import cho_inv
import prediction_formulae as pred 
import gp_tools
import test_functions as test_func 
import max_likelihood as max_llk




#Test for gp_tools
n = 100
xtest = 5*np.random.rand(n, 2)
theta_vec = [1, 1]
p = 2
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
sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
print(sighat)

#Test for mle
params = np.zeros((3, ))
params[0: 2] = theta_vec
params[2] = p
sigmle = max_llk.hat_sigmaz_sqr_mle(y, R, params)
print(sigmle)
llk = max_llk.log_likelihood(xtest, y, params)
print(llk)
xinit = np.array([100, 100, 1])
opti = max_llk.max_log_likelihood(xtest, y, xinit)
print(opti)



