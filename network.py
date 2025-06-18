import random
import numpy as np



class Network:
    def __init__(self, layer_sizes, vectorized = True):
        self.vectorized = vectorized
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    

    
    def feedforward(self, a): #calculate out of network given input a
        for weight, bias in zip(self.weights, self.biases):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None): #stochastic gradient descent given mini batch size and number of epochs
        if test_data:
            n_testdata = len(test_data)

        training_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, training_size, mini_batch_size)
            ]

            if(self.vectorized):
                for mini_batch in mini_batches:
                    self.update_vectorized(mini_batch, learning_rate)
            
            else:
                for mini_batch in mini_batches:
                    self.update_scalar(mini_batch, learning_rate)

            if test_data:
                print (f"Epoch {epoch}: {self.evaluate(test_data)} / {n_testdata}")

            else:
                print (f"Epoch {epoch} complete")


    def update_vectorized(self, mini_batch, learning_rate):
        X = np.hstack([x for x, y in mini_batch])
        Y = np.hstack([y for x, y in mini_batch])

        grad_w, grad_b = self.backprop_vectorized(X, Y)

        self.weights = [w - learning_rate * delta_w for w, delta_w in zip(self.weights, grad_w)]
        self.biases = [b - learning_rate * delta_b for b, delta_b in zip(self.biases, grad_b)]
        return

    def update_scalar(self, mini_batch, learning_rate):
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        for x, y in mini_batch:
            delta_grad_w, delta_grad_b = self.backprop_scalar(x, y)
            for i in range(len(self.weights)):
                #print(f"Layer {i}: weight shape = {self.weights[i].shape}, delta_grad_w shape = {delta_grad_w[i].shape}")
                grad_w[i] += delta_grad_w[i]
                grad_b[i] += delta_grad_b[i]

        #print("Norm of grad_w[0]:", np.linalg.norm(grad_w[0]))

        for i in range(len(self.weights)):
            #before = np.copy(self.weights[0])

            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]
            
            #print("Weight change:", np.linalg.norm(self.weights[0] - before))
        return

    def backprop_scalar(self, x, y): #backpropagation algorithm 
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        a = x
        activations = [a]
        z_values = []

        #run forward feed
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, a) + bias
            a = sigmoid(z)

            z_values.append(z)
            activations.append(a)
        
      #  for i, z in enumerate(z_values):
        #    print(f"Layer {i}: max z = {np.max(z)}, min z = {np.min(z)}")
        #start backwards run

        delta = [np.zeros_like(b) for b in self.biases]

        for layer in range(self.num_layers - 2, -1, -1):
           #print(f"Layer {layer}: delta = {delta[layer].shape}, activation = {activations[layer].shape}, grad_w = {delta[layer].shape} @ {activations[layer].T.shape}")

            if layer == self.num_layers - 2:
                delta[-1] = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_derivative(z_values[-1]))
            else:
                delta[layer] = np.multiply(np.matmul(self.weights[layer + 1].T, delta[layer + 1]), sigmoid_derivative(z_values[layer]))
            grad_w[layer] = np.matmul(delta[layer], np.transpose(activations[layer]))
            grad_b[layer] = delta[layer]
        return grad_w, grad_b
    
    def backprop_vectorized(self, X, Y): #backpropagation algorithm with X as a matrix containing all xs
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        activation = X
        activations = [X]
        z_values = []

        #run forward feed
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, activation) + bias
            activation = sigmoid(z)

            z_values.append(z)
            activations.append(activation)
        
      #  for i, z in enumerate(z_values):
        #    print(f"Layer {i}: max z = {np.max(z)}, min z = {np.min(z)}")
        #start backwards run

        batch_size = X.shape[1]
        delta = [np.zeros((b.shape[0], batch_size)) for b in self.biases]

        for layer in range(self.num_layers - 2, -1, -1):
           #print(f"Layer {layer}: delta = {delta[layer].shape}, activation = {activations[layer].shape}, grad_w = {delta[layer].shape} @ {activations[layer].T.shape}")
            if layer == self.num_layers - 2:
                delta[-1] = np.multiply(self.cost_derivative(activations[-1], Y), sigmoid_derivative(z_values[-1]))
            else:
                delta[layer] = np.multiply(np.matmul(self.weights[layer + 1].T, delta[layer + 1]), sigmoid_derivative(z_values[layer]))
            grad_w[layer] = np.matmul(delta[layer], np.transpose(activations[layer])) / batch_size
            grad_b[layer] = np.mean(delta[layer], axis = 1, keepdims=True)
        return grad_w, grad_b
    
    def evaluate(self, test_data): #evaluate accuracy based on validation set, specifically for mnist
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(predicted == actual) for (predicted, actual) in test_results)
    
    def cost_derivative(self, output_a, y):
        return output_a - y


def sigmoid(z): #sigmoid function
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))