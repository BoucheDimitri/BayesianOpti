import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as optimize
import prediction_formulae as pred
import exp_kernel


def fmin(y):
    return min(y)


def EI(xnew, xtest, y, Rinv, beta_hat, theta_vec, p_vec, function2Bmin):
        # Tu veux tout faire en meme temps mais c est pas une bonne idee,
        # il faut que tu puisses dire ce que fait chaque fonction exactement
        # La il y a des trucs que tu prends en argument, d'autres que tu recalcules,
        # Au final le role de la fonction est pas claire et c'est pas modulaire du tout
        # donc difficile a comprendre et a reutiliser
    f_min = fmin(y)
    y_hat = function2Bmin(xnew)
    rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
    sigma_hat = math.sqrt(pred.sigma_est(y, rx, Rinv, beta_hat))
    if sigma_hat == 0:
        EI = 0
    else:
        z = (f_min - y_hat) / sigma_hat
        EI = float((f_min - y_hat) * stats.norm.cdf(z) +
                   sigma_hat * stats.norm.pdf(z))
    print(EI)
    print(type(EI))
    return EI


def max_EI(xtest, y, Rinv, beta_hat, theta_vec, p_vec, xinit, function2Bmin):
    # bug when nb.it too large => ValueError: Objective function must return a
    # scalar

    # A priori n a pas sa place dans ce fichier, on veut regrouper toutes les fonctions d'acquisition
    # sous une forme unifiee pour pouvoir en changer facilement dans le reste du programme
    # Donc ce fichier ne doit contenir que des fonctions d'acquisition
    # On fera un fichier separe pour l'optimisation des fonctions d'acquisition
    def minus_EI(xnew): return float(-EI(xnew, xtest, y, Rinv,
                                         beta_hat, theta_vec, p_vec, function2Bmin))
    opti = optimize.minimize(
        fun=minus_EI, x0=xinit, bounds=(
            (0, 0), (5, 5)), method='SLSQP')
    return opti


def expected_improv(y, fmin, sigma):
    if sigma == 0:
        return 0.0
    else:
        z = (fmin - y) / sigma
        return (fmin - y) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
