import random
import numpy as np



class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    

    
    def feedforward(self, a): #calculate out of network given input a
        for weight, bias in zip(self.weights, self.biases):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, test_data = None): #stochastic gradient descent given mini batch size and number of epochs
        if test_data:
            n_testdata = len(test_data)

        training_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, training_size, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            if test_data:
                print (f"Epoch {epoch}: {self.evaluate(test_data)} / {n_testdata}")

            else:
                print (f"Epoch {epoch} complete")
                    
    def update_mini_batch(self, mini_batch, learning_rate):
        return

    def backprop(self, x, y): #backpropagation algorithm  
        return
    
    def evaluate(self): #evaluate accuracy based on validation set
        return
    
    def cost_derivative(self):
        return

def sigmoid(z): #sigmoid function
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(self, z):
    return sigmoid(z) * (1 - sigmoid(z))