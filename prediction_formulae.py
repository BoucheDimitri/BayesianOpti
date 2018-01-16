import numpy as np
import math
import cho_inv


#Pas sa place dans ce fichier
#Deplace dans le fichier gp_tools
#Sous le nom corr_func
#def SCF_exp(x1, x2, theta, p):
    #output = math.exp(theta[1])


def beta_est(y, R):
        # Tu inverses deux fois la matrice R alors que c'est une opération très couteuse (n^3)
        # Mieux de le faire une fois et de la stocker
        # Cf la version beta_est_cho ci-dessous
        # Et tu n'utilises pas le cholesky_inv
    n = y.shape[1]
    # Ca ne va pas marcher ici parce que numpy te créer un vecteur unidimensionnel
    # donc ca va merder dans le produit matriciel, il faut specifier un tuple
    # en shape
    ones = np.ones(n)
    beta_est = inv(
        ones.T.dot(
            inv(R)).dot(ones)).dot(
        inv(ones)).dot(
                inv(R)).dot(y)
    return beta_est


def beta_est_bis(y, Rinv):
    n = y.shape[1]
    ones = np.ones((n, 1))
    one_Rinv_one = float(np.dot(np.dot(ones.T, Rinv), ones))
    beta_est = (1.0 / one_Rinv_one) * np.dot(np.dot(ones.T, Rinv), y)
    return beta_est


#def beta_est_gene(y, Rinv, F):
    #Ft_Rinv = np.dot(F.T, Rinv)
    #Ft_Rinv_F = np.dot(Ft_Rinv, F)
    #Ft_Rinv_F_inv = cho_inv.cholesky_inv(Ft_Rinv_F)
    #beta est = np.dot(np.dot(Ft_Rinv_F_inv, Ft_Rinv), y)
    #return beta_est


def y_est(x, y, beta_hat, r_x, R):
        # Pareil si a chaque fois qu'on a besoin de l'inverse de R
        # dans une formule on le recalcule le computationnal overhead
        # Va être énomre, il faut mieux le passer en paramètre de la
        # Fonction comme ça on peut le faire qu'une fois par set de points de
        # données
    n = y.shape[1]
    y_est = beta_est + r(x).T.dot(inv(R)).dot(y - np.ones(n))
    return y_est


def y_est_bis(rx, y, Rinv, F, beta_est):
    n = y.shape[1]
    ones = np.ones((n, 1))
    rxt_Rinv = np.dot(rx.T, Rinv)
    y_est = beta_hat + np.dot(rxt_Rinv, y - beta_hat * ones)
    return y_est
