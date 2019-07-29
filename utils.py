import numpy as np
import matplotlib.pyplot as plt
import os
import struct

def read(dataset = "training", path = "."):
    """
    A python function for importing the MNIST data set. It returns two 
    numpy array, img and lbl, corresponding to images and labels.

    img.shape = (n, 28, 28)
    lbl.shape = (n, )
    """
        
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        print('Invalid input value')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype = np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype = np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl

def MNIST():
	Xtrain, ytrain = read('training', path = './MNIST')
	
	assert(Xtrain.shape == (60000, 28, 28))

	X = Xtrain.reshape((60000, 28 * 28))

	X = X.T

	return X

def fval(X, w):
	d, n = X.shape

	# A = np.dot(X, X.T)

	return np.linalg.norm(np.dot(X.T, w).squeeze()).item() ** 2 / n

def plot_obj_curve(X, w_list, plot = True):
	obj_list = []

	for i in range(len(w_list)):
		obj = fval(X, w_list[i])

		obj_list.append(obj)

	if plot == True:
		fig = plt.figure()
		plt.plot(obj_list)

	return obj_list

def plot_mse_curve(u, w_list):
	error_list = []

	for i in range(len(w_list)):
		error = np.linalg.norm(u - w_list[i])

		error_list.append(error)

	fig = plt.figure()
	plt.plot(error_list)

