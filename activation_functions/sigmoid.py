import numpy as np


class Sigmoid_Layer:
    def forward_pass(self, layer_inputs):
        self.layer_inputs = layer_inputs
        self.layer_output = 1 / (1 + np.exp(-layer_inputs))  # apply sigmoid formula

    def backward_pass(self, grad_inputs):
        self.grad_output = self.layer_output * (1 - self.layer_output) * grad_inputs  # derivative of sigmoid
