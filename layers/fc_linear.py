import numpy as np


class Fully_Connected_Layer:
    def __init__(self, neurons, output_units):
        self.output_units = output_units  # Output dimensions
        self.bias = np.zeros((1, output_units))  # initialize bias array
        self.weights = np.random.randn(neurons, output_units) * 0.05  # initialize weights with small scale values
        self.w_momentums = np.zeros_like(self.weights)  # initialize weight momentum array
        self.b_momentums = np.zeros_like(self.bias)  # initialize bias momentum array

    def forward_pass(self, layer_inputs):
        self.layer_inputs = layer_inputs  # save layer_inputs to self object used in backprop
        self.layer_output = np.dot(layer_inputs, self.weights) + self.bias  # dot product of layer inputs with weights

    def backward_pass(self, grad_inputs):
        self.grad_weights = np.dot(self.layer_inputs.T, grad_inputs)  # transpose values to fit the weights dimensions
        self.grad_bias = np.sum(grad_inputs, axis=0, keepdims=True)  # derivatives of weights and bias used to optimize the layer's weights and bias values
        self.grad_output = np.dot(grad_inputs, self.weights.T)  # transpose weights to fit neurons of layer dimensions
