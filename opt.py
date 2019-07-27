import numpy as np

def gd(X, w0 = None, eta = 0.01, epochs = 1):
	d, n = X.shape

	A = np.dot(X, X.T)

	A = A / n

	if w0 != None:
		w = w0
	else:
		w = np.random.randn(d, 1)
		w = w / np.linalg.norm(w)

	w_list = []
	w_list.append(w)

	for i in range(epochs):
		w = w + eta * np.dot(A, w)

		w = w / np.linalg.norm(w)

		w_list.append(w)

	return w, w_list

def sgd(X):
	assert(0)

def svrg(X):
	assert(0)


