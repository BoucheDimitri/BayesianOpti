import numpy as np
import math
import cho_inv

def SCF_exp(x1, x2, theta, p):
    output = math.exp(theta[1])

def beta_est(y, R)
    n = y.shape[1]
    ones = np.ones(n)
    beta_est = inv(ones.T.dot(inv(R)).dot(ones)).dot(inv(ones)).dot(inv(R)).dot(y)
    return beta_est


def y_est(x, y, beta_est, r_x, R):
    n = y.shape[1]
    y_est = beta_est+r(x).T.dot(inv(R)).dot(y-np.ones(n))
    return y_est
