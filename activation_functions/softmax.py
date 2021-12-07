import numpy as np


class Softmax_Layer:
    def forward_pass(self, layer_inputs):
        self.layer_inputs = layer_inputs
        exponential = np.exp(
            self.layer_inputs - np.max(self.layer_inputs, axis=1, keepdims=True))  # forward pass apply softmax formula
        likelihoods = exponential / np.sum(exponential, axis=1, keepdims=True)  # calculate softmax probabilities output
        self.layer_output = likelihoods  # save probabilities in self object as layer_output
