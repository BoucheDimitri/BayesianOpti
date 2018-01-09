import numpy as np


def cholesky_inv(a):
    """
    Compute the inverse of a symmetric matrix using Cholesky decomposition
    Args :
        a (numpy.ndarray) : the symmetric matrix to invert
    Returns :
        numpy.ndarray. The inverse of matrix a
    """
    l = np.linalg.cholesky(a)
    linv = np.linalg.inv(l)
    return np.dot(linv.T, linv)



#test = np.zeros((4, 4))
#test[0, 0] = 4
#test[0, 1] = 1.5
#test[0, 2] = 2.5
#test[0, 3] = 0.5
#test[1, 1] = 3
#test[1, 2] = 1.5
#test[1, 3] = 0.3
#test[2, 2] = 2
#test[2, 3] = 2.3
#test[3, 3] = 3
#test += test.T
#testinv = cholesky_inv(test)
#print(np.dot(test, testinv)