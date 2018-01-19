import numpy as np
import cho_inv
import prediction_formulae as pred 
import gp_tools
import test_functions as test_func 
import max_likelihood as max_llk
import acquisition_functions as af
import bayesian_optimization as bo


#Test for gp_tools
n = 10
xtest = np.random.rand(n, 2)
theta_vec = [1, 1]
p_vec = [1, 1]
#R = gp_tools.kernel_mat_2d(xtest, theta_vec, p_vec)
R = gp_tools.kernel_mat_2d_prod(xtest, theta_vec, p_vec)
print(R)
xnew = np.random.rand(2)
rx = gp_tools.kernel_rx_2d_prod(xtest, xnew, theta_vec, p_vec)
print(rx)



#Test for test_func
y = np.zeros((n, 1))
for i in range(0, n):
	y[i, 0] = test_func.mystery_vec(xtest[i, :])
print(y)



#Test for cho_inv
#Rinv = cho_inv.cholesky_inv(R)
#print(np.dot(Rinv, R))
Rinv = np.linalg.inv(R)


#Test for prediction_formulae
beta = pred.beta_est(y, Rinv)
print(beta)
y_hat = pred.y_est(rx, y, Rinv, beta)
print(y_hat)
sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
print(sighat)
sigma_sq_pred = pred.sigma_est(y, rx, Rinv, beta)
print(sigma_sq_pred)

#Test for mle
#params = np.zeros((4, ))
#params[0: 2] = theta_vec
#params[2: 4] = p_vec
#sigmle = max_llk.hat_sigmaz_sqr_mle(y, R)
#print(sigmle)
#llk = max_llk.log_likelihood(xtest, y, params)
#print(llk)
#xinit = np.array([0.2, 0.2])
#opti = max_llk.max_log_likelihood(xtest, y, xinit)
#print(opti)


#Test for mle
params = np.zeros((4, ))
params[0: 2] = theta_vec
params[2: 4] = p_vec
sigmle = max_llk.hat_sigmaz_sqr_mle(y, R)
print(sigmle)
llk = max_llk.log_likelihood(xtest, y, params)
print(llk)
xinit = np.array([0.2, 0.2])
opti = max_llk.max_log_likelihood(xtest, y, xinit)
print(opti)


#test for acquisition_functions
f_min = af.fmin(y)
print(f_min)
print(pred.sigma_est(y, rx, Rinv, beta))
print(test_func.mystery_vec(xnew))
ExpImp = af.EI(xnew, xtest, y, Rinv, beta, theta_vec, p_vec, test_func.mystery_vec)
print(ExpImp)
opti = af.max_EI(xtest, y, Rinv, beta, theta_vec, p_vec, np.random.rand(1,2), test_func.mystery_vec)
print(opti)
xnew = opti["x"]
xnew = xnew.reshape(1,2)
np.concatenate((xtest, xnew), axis=0)
ynew = opti["fun"].reshape(1,1)
np.concatenate((y, ynew))

#test for global optimization
n = 10
nb_it = 100
theta_vec = [1, 1]
p_vec = [1, 1]
min_gl, point_min = bo.bayesian_optimization(n, nb_it, p_vec, theta_vec, test_func.mystery_vec)
#math domain error...at sigma_hat = math.sqrt(pred.sigma_est(y, rx, Rinv, beta_hat)) ???