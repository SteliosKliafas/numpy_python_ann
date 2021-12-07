import collections
import numpy as np
from keras.datasets import mnist
from plot_functions.plts import plot_bar_chart


def data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()  # Load MNIST Datasets
    print("\n\n" + "-" * 20)
    print("| CHECKING DATA... |")
    print("-" * 20)
    counter = sorted(collections.Counter(train_y).items())  # Check if data of classification classes is balanced
    labels, values = zip(*counter)
    plot_bar_chart(labels, values)
    print("Check min and max values of data...... ", "MIN: ", train_x.min(), "MAX", train_x.max())
    print("Check data shape...... ", train_x.shape)
    train_x = (train_x.reshape(train_x.shape[0], -1).astype(np.float32)) / 255.0  # reshape data 60000, 784
    test_x = (test_x.reshape(test_x.shape[0], -1).astype(np.float32)) / 255.0  # and normalize values between -1 and 1
    print("Data reshaped...... ", train_x.shape,"\n\n")
    return (train_x, train_y), (test_x, test_y)