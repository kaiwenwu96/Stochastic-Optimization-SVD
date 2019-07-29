import numpy as np
from opt import *
from utils import *

if __name__ == '__main__':

    dataset = 'SYNTHETIC'
    # dataset = 'MNIST'

    if dataset == 'SYNTHETIC':
        d = 800
        n = 10000
    
        Sigma = 0.1 * np.eye(d)
        Sigma[0, 0] = 1
        Sigma[1, 1] = 0.50

        [Q1, R1] = np.linalg.qr(np.random.randn(d, d), mode = 'complete')
        [Q2, R2] = np.linalg.qr(np.random.randn(n, d), mode = 'reduced')

        X = np.dot(np.dot(Q1, Sigma), Q2.T)

    elif dataset == 'MNIST':
        X = MNIST()
        X = X[:, 0:500]
        X = X / 255

    d, n = X.shape 

    print('d = %d, n = %d' % (d, n))

    U, S, Vh = np.linalg.svd(X, full_matrices = False)

    # print('spectral norm of XX^T = %f' % (np.linalg.norm(np.dot(X, X.T) / n, ord = 2)))

    eigen_value = S[0] ** 2 / n

    print('Singular value = %f, %f, %f' % (S[0], S[1], S[2]))
    print('Eigen value = %f' % (eigen_value))

    epochs = 20

    # power iteration
    print('******** Power ********')

    w, w_list = powerIteration(X, epochs = epochs)

    print('Power eigen value = %f' % fval(X, w))
    power_val_list = plot_obj_curve(X, w_list, plot = False)

    # gradient descent
    print('******** GD ********')

    eta = 1.0 / eigen_value / 2.0
    print('GD eta = %f' % eta)

    w, w_list = gd(X, eta = eta, epochs = epochs)

    print('GD eigen value = %f' % fval(X, w))
    gd_val_list = plot_obj_curve(X, w_list, plot = False)

    # stochastic gradient descent
    print('******** SGD ********')

    eta = 1.0 / eigen_value
    print('SGD eta = %f' % eta)

    w, w_list = sgd(X, eta = eta, epochs = epochs)

    print('SGD eigen value = %f' % fval(X, w))
    sgd_val_list = plot_obj_curve(X, w_list, plot = False)

    # svrg
    print('******** SVRG ********')

    mean_norm = np.mean(np.linalg.norm(X, ord = 2, axis = 0) ** 2)
    eta = 1.0 / mean_norm / np.sqrt(n)

    print('SVRG eta = %f' % eta)

    w, w_list = svrg(X, eta = eta, epochs = epochs)

    print('SVRG eigen value %f' % fval(X, w))
    svrg_val_list = plot_obj_curve(X, w_list, plot = False)

    fig = plt.figure(figsize = (6, 5))
    
    # plt.subplot(1, 2, 1)

    plt.plot( - np.array(power_val_list), linewidth = 2, linestyle = '--', label = 'Power')
    plt.plot( - np.array(gd_val_list), linewidth = 2, linestyle = '--', label = 'GD')
    plt.plot( - np.array(sgd_val_list), linewidth = 2, linestyle = '--', label = 'SGD')
    plt.plot( - np.array(svrg_val_list), linewidth = 2, linestyle = '--', label = 'SVRG')
    plt.plot( - eigen_value * np.ones(len(sgd_val_list)), linewidth = 2, label = 'Minimum')
    
    plt.ylim(bottom = - eigen_value * 1.1)
    plt.legend(loc = 'upper right')
    plt.title('Error vs Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Error')

    # plt.subplot(1, 2, 2)

    fig = plt.figure(figsize = (6, 5))

    plt.plot(np.log10(- np.array(power_val_list) + eigen_value), linewidth = 2, label = 'Power')
    plt.plot(np.log10(- np.array(gd_val_list) + eigen_value), linewidth = 2, label = 'GD')
    plt.plot(np.log10(- np.array(sgd_val_list) + eigen_value), linewidth = 2, label = 'SGD')
    plt.plot(np.log10(- np.array(svrg_val_list) + eigen_value), linewidth = 2, label = 'SVRG')

    plt.ylim(bottom = - 12)
    plt.legend(loc = 'upper right')
    plt.title('Log Error vs Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Log Error')
    # plt.tick_params(labelsize = 24)   

    plt.tight_layout()

    plt.show()
