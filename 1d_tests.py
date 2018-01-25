import numpy as np
import matplotlib.pyplot as plt

import cho_inv
import exp_kernel
import test_functions as test_func
import max_likelihood as max_llk
import prediction_formulae as pred
import bayesian_optimization as bayes_opti
import acquisition_functions as af
import acquisition_max as am
import AcqFuncs as AF
import visualization as viz


# Test for gp_tools
n = 5
xtest = 10 * np.random.rand(n, 1)
theta_vec = np.array([1])
p_vec = np.array([1.5])
#R = gp_tools.kernel_mat_2d(xtest, theta_vec, p_vec)
R = exp_kernel.kernel_mat(xtest, theta_vec, p_vec)
xnew = np.random.rand(1)
rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
print(rx)

#Y
y = np.zeros((n, 1))
for i in range(0, n):
    y[i, 0] = test_func.test_1d(xtest[i, :])


#Plot objective and points
axes = viz.plot_func_1d((0, 10), 1000, test_func.test_1d, nsub_plots=2)
axes[0] = viz.add_points_1d(axes[0], xtest, y)



# Test for cho_inv
Rinv = cho_inv.cholesky_inv(R)
#print(np.dot(Rinv, R))
#Rinv = np.linalg.inv(R)


# Test for prediction_formulae
beta = pred.beta_est(y, Rinv)
#print(beta)
y_hat = pred.y_est(rx, y, Rinv, beta)
# print(y_hat)
sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
# print(sighat)
sigma_sq_pred = pred.sigma_sqr_est(y, rx, Rinv, beta)
# print(sigma_sq_pred)


fmin = np.min(y)
hat_sigma = np.power(sigma_sq_pred, 0.5)
exp_impr = AF.ExpImpr(xi=0.01, fmin=fmin)

axes[1] = viz.plot_acq_func_1d(xtest,
                     y,
                     Rinv,
                     beta,
                     theta_vec,
                     p_vec,
                     (0, 10),
                     1000,
                     exp_impr,
                     axis=axes[1])



xgrid = np.linspace(0, 10, 1000)
gp_means, gp_stds = pred.pred_means_stds(xgrid, xtest, y, theta_vec, p_vec)
axes[0].plot(xgrid, gp_means)
plt.show()

# nit = 5
#
# for i in range(0, nit):
#     exp_impr = AF.ExpImpr(xi=0, fmin=fmin)
#     print (exp_impr.evaluate(y_hat, hat_sigma))


# Test for bayesian optimization
# xbest = bayes_opti.bayesian_search(xtest, y, theta_vec, p_vec, xnew, exp_impr)
# print(xbest)
#
# n_it = 10
# xx, yy = bayes_opti.bayesian_opti(xtest, y, n_it,
#                                   theta_vec,
#                                   p_vec,
#                                   exp_impr,
#                                   test_func.test_1d,
#                                   bounds=[(0, 10)])
#
# print(xx)