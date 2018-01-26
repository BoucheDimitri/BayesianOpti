import numpy as np
import cho_inv
import exp_kernel
import test_functions as test_func
import prediction_formulae as pred
import AcqFuncs as AF
import visualization as viz


# Test for gp_tools
n = 5
xtest = 5 * np.random.rand(n, 2)
theta_vec = np.array([1, 1])
p_vec = np.array([1.5, 1.5])
#R = gp_tools.kernel_mat_2d(xtest, theta_vec, p_vec)
R = exp_kernel.kernel_mat(xtest, theta_vec, p_vec)
# print(R)
xnew = np.random.rand(2)
rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
# print(rx)
image = test_func.mystery_vec(xnew)


# Test for test_func
y = np.zeros((n, 1))
for i in range(0, n):
    y[i, 0] = test_func.mystery_vec(xtest[i, :])
# print(y)


# Test for cho_inv
Rinv = cho_inv.cholesky_inv(R)
#print(np.dot(Rinv, R))
#Rinv = np.linalg.inv(R)


# Test for prediction_formulae
beta = pred.beta_est(y, Rinv)
# print(beta)
y_hat = pred.y_est(rx, y, Rinv, beta)
# print(y_hat)
sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
# print(sighat)
sigma_sq_pred = pred.sigma_sqr_est(y, rx, Rinv, beta)
# print(sigma_sq_pred)


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
# Test for acquisition functions :
fmin = np.min(y)
hat_sigma = np.power(sigma_sq_pred, 0.5)

# EI
exp_impr = AF.ExpImpr(xi=0, fmin=fmin)
print (exp_impr.evaluate(y_hat, hat_sigma))

# LCB
low_conf_bound = AF.LowConfBound()
print (low_conf_bound.evaluate(y_hat, hat_sigma))
#
# # Test acquisition_max
# test = am.complete_acq_func(
#     xtest,
#     xnew,
#     y,
#     Rinv,
#     beta,
#     theta_vec,
#     p_vec,
#     exp_impr)
#
# print(test)
#
# # Test for max acq_func
# opt = am.opti_acq_func(
#     xtest,
#     y,
#     Rinv,
#     beta,
#     theta_vec,
#     p_vec,
#     xnew,
#     low_conf_bound)
#     #bounds=((0, 0), (5, 5)))
# print(opt)


# Test for bayesian optimization
# xbest = bayes_opti.bayesian_search(xtest, y, theta_vec, p_vec, xnew, exp_impr)
# print(xbest)
#
# n_it = 10
# xx, yy = bayes_opti.bayesian_opti(xtest, y, n_it,
#                                   theta_vec,
#                                   p_vec,
#                                   exp_impr,
#                                   test_func.mystery_vec,
#                                   bounds=((0, 5), (0, 5)))
# print(xx.shape)
# print(np.min(yy))


viz.plot_acq_func_2d(xtest,
                     y,
                     Rinv,
                     beta,
                     theta_vec,
                     p_vec,
                     ((0, 5), (0, 5)),
                     (100, 100),
                     exp_impr)

viz.plot_func_2d(((0, 5), (0, 5)), (50, 50), test_func.mystery_vec)
