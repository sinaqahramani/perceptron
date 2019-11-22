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