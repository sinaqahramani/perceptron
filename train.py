# Importing moduls:
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from perceptron import Perceptron

# Loading data:
np.random.seed(42)

test_data, train_data_list, test_labels, train_labels = load_data('data/')
train_data = concat(train_data_list)

print(train_data.shape)  # Expected ouput: (60000, 784)
print(test_data.shape)  # Expected ouput: (10000, 784)
print(test_labels.shape)  # Expected ouput: (10000,)
print(train_labels.shape)  # Expected ouput: (60000,)

train_data_shuffled, train_labels_shuffled = shuffle(train_data, train_labels)
X_test, y_test = test_data, test_labels
X_train, X_val = split_train(train_data, val_ratio=1/60)
print(X_train.shape, X_val.shape)  # Expected ouput: (59000, 784) (1000, 784)
y_train, y_val = split_train(train_labels, val_ratio=1/60)
print(y_train.shape, y_val.shape)  # Expected ouput: (59000,) (1000,)

# adding 1s to the end of feature vectors to be multiplied by bias term of weights
X_val = np.insert(X_val, 0, 1, axis=1)
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)
print(X_train.shape)  # Expected ouput: (59000, 785)
print(X_val.shape)  # Expected ouput: (1000, 785)
print(X_test.shape)  # Expected ouput: (10000, 785)

# Making model using Perceptron class
model = Perceptron(784, 10)

# start training:
val_accs = []
for i, (x, y) in enumerate(zip(X_train, y_train)):
    model.train(x, y)
    if i % 1000 == 0:
        val_res =  [model.predict(x_val) == y_val for x_val, y_val in zip(X_val, y_val)]
        val_acc = np.sum(val_res) / len(val_res)
        val_accs.append(val_acc*100)  # recording the accuray to be plotted after training 
        # verbose:
        print("iteration number %d, accuracy on validation set: %.2f%%" % (i, 100*val_acc))

# test:
test_res =  [model.predict(x_test) == y_test for x_test, y_test in zip(X_test, y_test)]
test_acc = np.sum(test_res) / len(test_res)
print("-" * 60)
print("accuracy on test set: %.2f%%" % (100*test_acc))

plt.plot(val_accs)
plt.ylabel('validation accuracy')
plt.xlabel('iteration number')
plt.show()