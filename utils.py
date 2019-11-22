# Importing modules:
import numpy as np
import os

# Load data function:
def load_data(root_path):
    """
    root_path: the root directory of data files.
    outputs:
    1) test_data: test data which is a numpy array of shape (10000, 784).
    2) train_data_array: a Python list containing 5 numpy 2d-arrays each one of them has the shape (12000, 784).
    3) test_labels: a numpy array with shape (10000, ) containing the labels (0 to 9) of the test set images.
    4) train_labels: a numpy array with shape (60000, ) containing the labels (0 to 9) of the training set images.
    """
    train_data_file_names = ['train{}.npy'.format(x) for x in range(1, 6)]
    test_data_file_name = 'test.npy'
    test_labels_file_name = 'labels_test.npy'
    train_labels_file_name = 'labels_train.npy'
    
    train_data_list = []
    for file_name in train_data_file_names:
        file_path = os.path.join(root_path, file_name)
        loaded = np.load(file_path)
        train_data_list.append(loaded)
        
    test_data = np.load(os.path.join(root_path, test_data_file_name))
    test_labels = np.load(os.path.join(root_path, test_labels_file_name))
    train_labels = np.load(os.path.join(root_path, train_labels_file_name))
    
    return test_data, train_data_list, test_labels, train_labels
	
# concat function:
def concat(list_of_arrays):
    """
    list_of_arrays is a non-empty list of numpy arrays (with arbitrary number of dimensions) with the SAME shape
    output: array_concat which is a numpy ndarray
    """
    shape = np.shape(list_of_arrays)
    newShape = [ shape[0]*shape[1] ]
    if len(shape)>2:
      for i in range(2,len(shape)):
        newShape.append(shape[i])
    
    array_concat = np.zeros(newShape)
    s=0
    e=shape[1]
    
    for i in range(0,shape[0]):
      array_concat[s:e] = list_of_arrays[i]
      s=e
      e=e+shape[1]   
    return array_concat
	
# reshape function:
def reshape(x):
    """
    x is an array with shape (N, d*d)
    output: x_reshaped which is numpy 3d-array with shape (N, d, d)
    """
    shape = np.shape(x)
    d = np.sqrt(shape[1])
    d = d.astype(np.int32)
    x_reshaped = np.reshape( x , (shape[0],d,d) )
    return x_reshaped
	
# split train function
def split_train(x, val_ratio=0.1):
    """
    x: a numpy ndarray
    val_ratio: ratio between the size of validation set and x
    ouputs: x_train, x_val: numpy ndarrays
    """
    shape = x.shape
    ind = np.int32((shape[0])*val_ratio)
    ind =shape[0]-ind
    x_train = x[:ind]
    x_val = x[ind:]
    return x_train, x_val	

# shuffle function:
def shuffle(x, y):
    """
    x: a numpy nd-array with shape (N, d1, d2, ...)
    y: a numpy 1d-array with shape (N, )
    output: x_shuffled and y_shuffled
    """
    shape = x.shape
    newInd = np.arange(0,shape[0])
    np.random.shuffle(newInd)
    x_shuffled, y_shuffled = x[newInd],y[newInd]   
    return x_shuffled, y_shuffled