import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size): # input is expected to be col vector
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_grad, learning_rate):
        # calculating gradeints w.r.t W, B, & X
        weights_gradient = np.dot(output_grad, self.input.T)
        bias_gradient = output_grad
        input_gradient = np.dot(self.weights.T, output_grad)
        # optimizing weights & bias
        self.weights -= learning_rate*weights_gradient
        self.bias -= learning_rate*bias_gradient
        return input_gradient