import numpy as np
import scipy.misc as misc
import math
import scipy.stats as stats
import scipy.optimize as optimize
import prediction_formulae as pred
import exp_kernel


def fmin(y):
    return min(y)


def EI(xnew, xtest, y, Rinv, beta_hat, theta_vec, p_vec, function2Bmin):

        # Si on prend ta logique de code, on va devoir recoder toutes les
        # Fonctions pour chaque fonction d'acquisition, clairement sous optimal
        # Cf mon fichier acquisition max
        # Reprend sa logique quand tu t'y remettras

    f_min = fmin(y)

    # Ici ce n'est pas la bonne formule pour y_hat puisque
    # meme si ce n'est pas vraiment le cas ici, l'article traite
    # de la minimization de "expensive" functions
    # Donc c'est un peu de la triche d'evaluer la fonction
    # A chaque nouveau point que l'on se propose...
    # Le role de l'expected improvement est justement de savoir ou on
    # va evaluer la fonction pour eviter l'evaluation explicite
    # Donc il faut reprendre la formule de l'article pour l'estimation de
    # y(xnew)
    y_hat = function2Bmin(xnew)
    rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
    sigma_hat = math.sqrt(pred.sigma_sqr_est(y, rx, Rinv, beta_hat))
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
    # sous une forme "unifiee" pour pouvoir en changer facilement dans le reste du programme
    # Donc ce fichier ne doit contenir que des fonctions d'acquisition
    # On fera un fichier separe pour l'optimisation des fonctions d'acquisition

    # D'autre part avec cette logique de code on va devoir reecrire toutes les fonctions
    # Pour chacune des fonctions d'acquisition ce qui est plutot sous optimal
    # Cf mon fichier acquisition_max et les fonction d'acquisition ci-apres
    def minus_EI(xnew): return float(-EI(xnew, xtest, y, Rinv,
                                         beta_hat, theta_vec, p_vec, function2Bmin))
    opti = optimize.minimize(
        fun=minus_EI, x0=xinit, bounds=(
            (0, 0), (5, 5)), method='SLSQP')
    return opti


def expected_improvement(hat_y, hat_sigma, xi, fmin):
    if hat_sigma == 0:
        return 0.0
    else:
        z = (fmin - hat_y - xi) / hat_sigma
        value = (fmin - hat_y - xi) * stats.norm.cdf(z) + \
            hat_sigma * stats.norm.pdf(z)
        print(value)
        return max(0, value)


def lower_confidence_bound(hat_y, hat_sigma, xi):
    return max(0, hat_y - xi * hat_sigma)


def bandit_lower_confidence_bound(hat_y, hat_sigma, xi, delta, t, d):
    tau_t = 2 * np.log(np.power(float(d)/2.0 + 2, t) * np.power(np.pi, 2) / (3 * delta))
    return max(0, hat_y - np.sqrt(xi * tau_t) * hat_sigma)


def g_expected_improvement(hat_y, hat_sigma, xi, fmin, g):
    # Generalized expected improvement, runs
    # But obviously erroneous results
    if hat_sigma == 0:
        return 0.0
    else:
        z = (fmin - hat_y - xi) / hat_sigma
        t0 = stats.norm.cdf(z)
        t1 = - stats.norm.pdf(z)
        s = 0
        for k in range(0, g):
            if k / 2 == 0:
                s += misc.comb(g, k) * np.power(z, g - k) * t0
                if k != 0:
                    t0 = - stats.norm.pdf(z) * \
                        np.power(z, k - 1) + (k - 1) * t0
            else:
                s -= misc.comb(g, k) * np.power(z, g - k) * t1
                if k != 1:
                    t1 = - stats.norm.pdf(z) * \
                        np.power(z, k - 1) + (k - 1) * t1
    return max(0, np.power(hat_sigma, g) * s)


# Dictionnary of acquisition functions
acq_funcs_dic = dict()
acq_funcs_dic["EI"] = expected_improvement
acq_funcs_dic["LCB"] = lower_confidence_bound
acq_funcs_dic["GEI"] = g_expected_improvement
