import numpy as np

def powerIteration(X, w0 = None, epochs = 1):
	d, n = X.shape

	# A = np.dot(X, X.T)

	# A = A / n

	if w0 != None:
		w = w0
	else:
		w = np.random.randn(d, 1)
		w = w / np.linalg.norm(w)

	w_list = []
	w_list.append(w)

	for i in range(epochs):
		# w = np.dot(A, w)
		w = 1.0 / n * np.dot(X, np.dot(X.T, w))

		w = w / np.linalg.norm(w)

		w_list.append(w)

	return w, w_list

def gd(X, w0 = None, eta = 0.01, epochs = 1):
	d, n = X.shape

	# A = np.dot(X, X.T)

	# A = A / n

	if w0 != None:
		w = w0
	else:
		w = np.random.randn(d, 1)
		w = w / np.linalg.norm(w)

	w_list = []
	w_list.append(w)

	for i in range(epochs):
		# w = w + eta * np.dot(A, w)
		w = w + eta / n * np.dot(X, np.dot(X.T, w))

		w = w / np.linalg.norm(w)

		w_list.append(w)

	return w, w_list

def sgd(X, w0 = None, eta = 0.01, epochs = 1):
	d, n = X.shape

	if w0 != None:
		w = w0
	else:
		w = np.random.randn(d, 1)
		w = w / np.linalg.norm(w)

	w_list = []
	w_list.append(w)

	for i in range(epochs):
		for j in range(n):
			it = np.random.choice(n)

			x = X[:, it].reshape((d, 1))
		
			w = w + eta / (i * d + j + 1) * np.dot(x.T, w).item() * x

			w = w / np.linalg.norm(w)

		w_list.append(w)

	return w, w_list

def svrg(X, w0 = None, eta = 0.01, epochs = 1):
	d, n = X.shape

	if w0 != None:
		ws = w0
	else:
		ws = np.random.randn(d, 1)
		ws = ws / np.linalg.norm(ws)

	w_list = []
	w_list.append(ws)

	for i in range(epochs):
		u = 1.0 / n * np.dot(X, np.dot(X.T, ws))
		wt = ws
		
		for j in range(n):
			it = np.random.choice(n)

			x = X[:, it].reshape((d, 1))

			wt = wt + eta * (x * (np.dot(x.T, wt) - np.dot(x.T, ws)) + u) 

			# print(wt)

			wt = wt / np.linalg.norm(wt)
		
		ws = wt

		w_list.append(ws)

	return ws, w_list
