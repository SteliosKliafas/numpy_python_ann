from activation_functions.softmax import Softmax_Layer
from loss.cross_entropy_loss import CrossEntropyLoss


class Softmax_CrossEntropy_Loss_Layer:
    def __init__(self):
        self.softmax = Softmax_Layer()  # creating instance of softmax layer class
        self.loss = CrossEntropyLoss()  # creating instance of crossentropy loss class

    def forward_pass(self, layer_inputs, y_true):
        self.softmax.forward_pass(layer_inputs)  # get probabilities of softmax
        self.layer_output = self.softmax.layer_output  # save probabilities to self
        return self.loss.forward_pass(self.layer_output, y_true)  #  calculate loss with cross entropy by passing softmax probabilities

    def backward_pass(self, grad_inputs, y_true):
        self.grad_output = self.loss.backward_pass(grad_inputs, y_true)  # back propagate through cross entropy and softmax combined formula