import numpy as np
from opt import *
from utils import *

if __name__ == '__main__':
	d = 1000
	n = 10000

	X = np.random.randn(d, n)

	X = MNIST()
	X = X / 255

	d, n = X.shape 

	U, S, Vh = np.linalg.svd(X, full_matrices = False)

	# print('spectral norm of XX^T = %f' % (np.linalg.norm(np.dot(X, X.T) / n, ord = 2)))

	eigen_value = S[0] ** 2 / n
	print('Eigen value = %f' % (eigen_value))

	epochs = 20

	# power iteration
	w, w_list = powerIteration(X, epochs = epochs)

	print('Power eigen value = %f' % fval(X, w))
	power_val_list = plot_obj_curve(X, w_list, plot = False)

	# gradient descent
	eta = 2.0 / eigen_value
	print('GD eta = %f' % eta)

	w, w_list = gd(X, eta = eta, epochs = epochs)

	print('GD eigen value = %f' % fval(X, w))
	gd_val_list = plot_obj_curve(X, w_list, plot = False)

	# stochastic gradient descent
	eta = 2.0 * 2.0 / eigen_value
	print('SGD eta = %f' % eta)

	w, w_list = sgd(X, eta = eta, epochs = epochs)

	print('SGD eigen value = %f' % fval(X, w))
	sgd_val_list = plot_obj_curve(X, w_list, plot = False)

	# svrg
	mean_norm = np.mean(np.linalg.norm(X, ord = 2, axis = 0))
	eta = 1.0 / mean_norm / np.sqrt(n)

	print('SVRG eta = %f' % eta)

	w, w_list = svrg(X, eta = eta, epochs = epochs)

	print('SVRG eigen value %f' % fval(X, w))
	svrg_val_list = plot_obj_curve(X, w_list, plot = False)

	fig = plt.figure()
	plt.plot( - np.array(power_val_list), linewidth = 2, label = 'Power')
	plt.plot( - np.array(gd_val_list), linewidth = 2, label = 'GD')
	plt.plot( - np.array(sgd_val_list), linewidth = 2, label = 'SGD')
	plt.plot( - np.array(svrg_val_list), linewidth = 2, label = 'SVRG')
	plt.plot( - eigen_value * np.ones(len(sgd_val_list)), linewidth = 2, linestyle = '--', label = 'Minimum')
	plt.legend(loc = 'upper right')

	fig = plt.figure()
	plt.plot(np.log10(- np.array(power_val_list) + eigen_value), linewidth = 2, label = 'Power')
	plt.plot(np.log10(- np.array(gd_val_list) + eigen_value), linewidth = 2, label = 'GD')
	plt.plot(np.log10(- np.array(sgd_val_list) + eigen_value), linewidth = 2, label = 'SGD')
	plt.plot(np.log10(- np.array(svrg_val_list) + eigen_value), linewidth = 2, label = 'SVRG')
	plt.legend(loc = 'upper right')

	plt.show()
