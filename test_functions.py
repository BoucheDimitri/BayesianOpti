import math


def mystery(x1, x2):
    #could be good to add a little noise no?
	"""
    Test function for optimization

    Args:
        x1 (float) : 0 <= x1 <= 5
        x2 (float) : 0 <= x2 <= 5

    Returns:
        float. mystery_function(x1, x2)
	"""
	a = 0.01 * (x2 - x1 * x1) * (x2 - x1 * x1)
	b = 2 * (2 - x2) * (2 - x2)
	c = 7 * math.sin(0.5 * x1) * math.sin(0.7 * x1 * x2)
	return 2 + a + b + c


def mystery_vec(xvec):
	"""
	mystery function taking vector as input

	Args:
	    xvec (numpy.ndarray) : shape = (2, )
	"""
	return mystery(xvec[0], xvec[1])

# The global solution has a value of -1.4565 at x = [2.5044,2.5778]