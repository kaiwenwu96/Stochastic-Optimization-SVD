import numpy as np
from opt import *
from utils import *

if __name__ == '__main__':
	d = 100
	n = 20

	X = np.random.randn(d, n)
	# X = np.zeros((d, n))

	U, S, Vh = np.linalg.svd(X)

	# print('spectral norm of XX^T = %f' % (np.linalg.norm(np.dot(X, X.T) / n, ord = 2)))

	eigen_value = S[0] ** 2 / n
	print('Eigen Value = %f' % (eigen_value))

	epochs = 100

	# power iteration
	w, w_list = powerIteration(X, epochs = epochs)

	print(fval(X, w))
	power_val_list = plot_obj_curve(X, w_list, plot = False)

	# gradient descent
	w, w_list = gd(X, eta = 0.01, epochs = epochs)

	print(fval(X, w))
	gd_val_list = plot_obj_curve(X, w_list, plot = False)

	# stochastic gradient descent
	w, w_list = sgd(X, eta = 0.5, epochs = epochs)

	print(fval(X, w))
	sgd_val_list = plot_obj_curve(X, w_list, plot = False)

	# svrgk
	w, w_list = svrg(X, eta = 0.01, epochs = epochs)

	print(fval(X, w))
	svrg_val_list = plot_obj_curve(X, w_list, plot = False)

	# fig = plt.figure()
	# plt.plot(np.log10(1 - power_val_list / eigen_value), linewidth = 2, label = 'Power')
	# plt.plot(np.log10(1 - gd_val_list / eigen_value), linewidth = 2, label = 'GD')
	# plt.plot(np.log10(1 - sgd_val_list / eigen_value), linewidth = 2, label = 'SGD')
	# plt.plot(np.log10(1 - svrg_val_list / eigen_value), linewidth = 2, label = 'SVRG')
	# # plt.plot( - eigen_value * np.ones(len(sgd_val_list)), linewidth = 2, linestyle = '--', label = 'Minimum')
	# plt.legend(loc = 'upper right')

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
	# plt.plot( - eigen_value * np.ones(len(sgd_val_list)), linewidth = 2, linestyle = '--', label = 'Minimum')
	plt.legend(loc = 'upper right')

	plt.show()
