import numpy as np
import math
import cho_inv

def SCF_exp(x1, x2, theta, p):
    output = math.exp(theta[1])

def beta_est(y, R):
	#Tu inverses deux fois la matrice R alors que c'est une opération très couteuse (n^3)
	#Mieux de le faire une fois et de la stocker
	#Cf la version beta_est_cho ci-dessous
	#Et tu n'utilises pas le cholesky_inv 
    n = y.shape[1]
    #Ca ne va pas marcher ici parce que numpy te créer un vecteur unidimensionnel
    #donc ca va merder dans le produit matriciel, il faut specifier un tuple en shape
    ones = np.ones(n)
    beta_est = inv(ones.T.dot(inv(R)).dot(ones)).dot(inv(ones)).dot(inv(R)).dot(y)
    return beta_est


def beta_est_cho(y, R):
	#Beta qui sors est unidimensionnel si l'on suit la formule
	#Donnee dans l'overleaf, est-ce normal ?
    n = y.shape[1]
    Rinv = cho_inv.cholesky_inv(R)
    ones = np.ones((n, 1))
    one_Rinv_one = float(np.dot(np.dot(ones.T, Rinv), ones))
    beta_est = (1.0/one_Rinv_one) * np.dot(np.dot(ones.T, Rinv), y)
    return beta_est



def y_est(x, y, beta_est, r_x, R):
    n = y.shape[1]
    y_est = beta_est+r(x).T.dot(inv(R)).dot(y-np.ones(n))
    return y_est
