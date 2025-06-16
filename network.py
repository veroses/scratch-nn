import random
import numpy as np



class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    

    
    def feedforward(self, a): #calculate out of network given input a
        for weight, bias in zip(self.weights, self.biases):
            a = sigmoid(np.matmul(weight, a) + bias)
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
                    
    def update(self, mini_batch, learning_rate):
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        for x, y in mini_batch:
            delta_grad_w, delta_grad_b = self.backprop(x, y)
            for i in range(self.num_layers - 1):
                grad_w[i] += delta_grad_w[i]
                grad_b[i] += delta_grad_b[i]

        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]
        

    def backprop(self, x, y): #backpropagation algorithm  
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        activations = [x]
        z_values = []

        #run forward feed
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, x) + bias
            a = sigmoid(z)

            z_values.append(z)
            activations.append(a)
        
        #start backwards run

        delta = [np.zeros_like(b) for b in self.biases]

        for layer in range(self.num_layers - 2, -1, -1):
            if layer == self.num_layers - 2:
                delta[-1] = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_derivative(z_values[-1]))
            else:
                delta[layer] = np.multiply(np.matmul(self.weights[layer + 1], delta[layer + 1]), sigmoid_derivative(z_values[layer]))
            grad_w[layer] = np.matmul(delta[layer], np.transpose(activations[layer - 1]))
            grad_b[layer] = delta[layer]
        return grad_w, grad_b
    
    def evaluate(self, test_data): #evaluate accuracy based on validation set, specifically for mnist
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(predicted == actual) for (predicted, actual) in test_results)
    
    def cost_derivative(self, output_a, y):
        return output_a - y

def sigmoid(z): #sigmoid function
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(self, z):
    return sigmoid(z) * (1 - sigmoid(z))