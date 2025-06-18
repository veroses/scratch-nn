from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np


def one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]

def my_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize and flatten
    x_train = x_train.reshape((-1, 28*28)) / 255.0
    x_test  = x_test.reshape((-1, 28*28)) / 255.0
   
    y_train_oh = one_hot(y_train)

    x_validate = x_train[50000:]
    x_train_f = x_train[:50000]
    y_validate = y_train_oh[50000:]
    y_train_f = y_train_oh[:50000]

    x_test = [x.reshape(784, 1) for x in x_test]

    train_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_train_f, y_train_f)]
    test_data = [(x.reshape(-1, 1), y) for x, y in zip(x_test, y_test)]
    val_data = [(x.reshape((-1, 1)), y.reshape(-1, 1)) for x, y in zip(x_validate, y_validate)]
    return train_data, test_data, val_data