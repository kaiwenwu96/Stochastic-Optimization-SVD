import numpy as np
from opt import *
from utils import *

if __name__ == '__main__':
	d = 100
	n = 200

	X = np.random.randn(d, n)
	# X = np.zeros((d, n))

	U, S, Vh = np.linalg.svd(X)

	# print('spectral norm of XX^T = %f' % (np.linalg.norm(np.dot(X, X.T) / n, ord = 2)))

	w, w_list = gd(X, eta = 0.01, epochs = 1000)

	u = U[:, 0].reshape((d, 1))

	# print(u)
	# print(w_list[0])
	# print(w)
	# print(np.linalg.norm(w - u))
	print(fval(X, w))
	plot_obj_curve(X, w_list)
