import numpy as np
import math
import cho_inv


def beta_est(y, Rinv):
    """
    Estimate of beta taking Rinv as input (no inversion inside function)
    In simple kriging setup (so beta is a float)

    Args:
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : Inverse of R, shape=(n, n)

    Returns:
        float. Estimation of beta
    """
    n = y.shape[0]
    ones = np.ones((n, 1))
    #print(np.dot(ones.T, Rinv).shape)
    one_Rinv_one = float(np.dot(np.dot(ones.T, Rinv), ones))
    beta_est = (1.0 / one_Rinv_one) * np.dot(np.dot(ones.T, Rinv), y)
    return float(beta_est)


def y_est(rx, y, Rinv, beta_hat):
    n = y.shape[0]
    ones = np.ones((n, 1))
    rxt_Rinv = np.dot(rx.T, Rinv)
    y_est = beta_hat + np.dot(rxt_Rinv, y - beta_hat * ones)
    return float(y_est)


def hat_sigmaz_sqr(y, Rinv, beta_hat):
    n = Rinv.shape[0]
    ones = np.ones((n, 1))
    err = y - beta_hat * ones
    return float((1.0 / float(n)) * np.dot(np.dot(err.T, Rinv), err))


def sigma_sqr_est(y, rx, Rinv, beta_hat):
    """
    Uses approximation formula 3.16 and replace sigma_z by 
    its MLE estimate from 3.17 to compute estimation of sigma
    """
    sigz_sqr = hat_sigmaz_sqr(y, Rinv, beta_hat)
    rxt_Rinv_rx = float(np.dot(np.dot(rx.T, Rinv), rx))
    return sigz_sqr * (1 - rxt_Rinv_rx)
