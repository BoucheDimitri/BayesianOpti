import math


def mystery_function(x1, x2):
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
