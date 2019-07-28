import numpy as np
import matplotlib.pyplot as plt

def fval(X, w):
	d, n = X.shape

	A = np.dot(X, X.T)

	return np.dot(np.dot(w.T, A), w).item() / n

def plot_obj_curve(X, w_list, plot = True):
	obj_list = []

	for i in range(len(w_list)):
		obj = - fval(X, w_list[i])

		obj_list.append(obj)

	if plot == True:
		fig = plt.figure()
		plt.plot(obj_list)
		plt.yscale(value = 'symlog')

	return obj_list
	
def plot_mse_curve(u, w_list):
	error_list = []

	for i in range(len(w_list)):
		error = np.linalg.norm(u - w_list[i])

		error_list.append(error)

	fig = plt.figure()
	plt.plot(error_list)

