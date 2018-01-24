import numpy as np
import cho_inv
import exp_kernel
import test_functions as test_func
import max_likelihood as max_llk
import prediction_formulae as pred
import bayesian_optimization as bayes_opti
import acquisition_functions as af
import acquisition_max as am


# Test for gp_tools
n = 100
xtest = 5 * np.random.rand(n, 2)
theta_vec = np.array([1, 1])
p_vec = np.array([0.5, 0.5])
#R = gp_tools.kernel_mat_2d(xtest, theta_vec, p_vec)
R = exp_kernel.kernel_mat(xtest, theta_vec, p_vec)
#print(R)
xnew = np.random.rand(2)
rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
#print(rx)
image = test_func.mystery_vec(xnew)


# Test for test_func
y = np.zeros((n, 1))
for i in range(0, n):
    y[i, 0] = test_func.mystery_vec(xtest[i, :])
#print(y)


# Test for cho_inv
Rinv = cho_inv.cholesky_inv(R)
#print(np.dot(Rinv, R))
#Rinv = np.linalg.inv(R)


# Test for prediction_formulae
beta = pred.beta_est(y, Rinv)
#print(beta)
y_hat = pred.y_est(rx, y, Rinv, beta)
#print(y_hat)
sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
#print(sighat)
sigma_sq_pred = pred.sigma_sqr_est(y, rx, Rinv, beta)
#print(sigma_sq_pred)


# # Test for mle
# params = np.zeros((4, ))
# params[0: 2] = theta_vec
# params[2: 4] = p_vec
# sigmle = max_llk.hat_sigmaz_sqr_mle(y, R)
# print(sigmle)
# llk = max_llk.log_likelihood(xtest, y, params)
# print(llk)
# params_init = np.array([0.5, 0.5, 1.5, 1.5])
# # Upper and lower bounds for optimization
# # If fixed_p set to False, useful to avoid convergence to singular matrix
# # leading to math errors
# mins_list = [None, None, 0, 0]
# maxs_list = [None, None, 1.99, 1.99]
# mle_opti = max_llk.max_log_likelihood(
#     xtest,
#     y,
#     params_init,
#     fixed_p=True,
#     mins_list=mins_list,
#     maxs_list=maxs_list)
# print(mle_opti)
#
#
# # Test for acquisition functions :
# fmin = np.min(y)
# hat_sigma = np.power(sigma_sq_pred, 0.5)
# print("EI")
# print(af.expected_improvement(y_hat, hat_sigma, 0, fmin))
# print("GEI 1")
# print(af.g_expected_improvement(y_hat, hat_sigma, 0, fmin, 1))
# print("GEI 5")
# print(af.g_expected_improvement(y_hat, hat_sigma, 0, fmin, 5))
# print("LCB")
# print(af.lower_confidence_bound(y_hat, hat_sigma, 1.5))
#
#
# # Test acquisition_max
# test = am.acq_func(
#     xtest,
#     xnew,
#     y,
#     Rinv,
#     beta,
#     theta_vec,
#     p_vec,
#     xi=0,
#     func_key="EI",
#     g=None)
# print(test)
#
# # Test for max acq_func
# opt = am.max_acq_func(
#     xtest,
#     y,
#     Rinv,
#     beta,
#     theta_vec,
#     p_vec,
#     xnew,
#     bounds=None,
#     xi=0,
#     func_key="EI",
#     g=None)
# print(opt)



#Test for bayesian optimization
xbest = bayes_opti.bayesian_search(xtest, y, theta_vec, p_vec, xnew, func_key="LCB")
#print(xbest)

print(xtest.shape)
print(xtest)
n_it = 50
xx, yy = bayes_opti.bayesian_opti(xtest, y, n_it,
                  theta_vec,
                  p_vec,
                  bounds=((0, 5), (0, 5)),
                  xi=0,
                  acq_func_key="EI",
                  test_func_key="Mystery")

print(xx.shape)
print(xx)
print(yy)
