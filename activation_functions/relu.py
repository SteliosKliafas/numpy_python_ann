import numpy as np


class ReLU_Layer:
    def forward_pass(self, layer_inputs):
        self.layer_inputs = layer_inputs
        self.layer_output = np.maximum(layer_inputs, 0)  # replace negative elements with 0

    def backward_pass(self, grad_inputs):
        self.grad_output = grad_inputs.copy()
        self.grad_output[self.layer_inputs <= 0] = 0  # derivative of ReLU