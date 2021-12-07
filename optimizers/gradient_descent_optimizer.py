class MiniBatch_GradientDescent_Optimizer:
    def __init__(self, lr=1, lr_decay=0, momentum=0):
        # initiate and store parameters of optimizer to self
        self.lr = lr
        self.lr_decay = lr_decay
        self.loop_number = 0
        self.momentum = momentum

    def update_learning_rate(self):
        self.lr = self.lr * (1 / (1 + self.lr_decay * self.loop_number))  # update the learning rate

    def update_weights_bias(self, layer):
        change_w = self.momentum * layer.w_momentums - self.lr * layer.grad_weights  # change of weights
        layer.w_momentums = change_w  # update weight momentums equal to the change of weights

        change_b = self.momentum * layer.b_momentums - self.lr * layer.grad_bias  # change of bias values
        layer.b_momentums = change_b  # update bias momentums equal to the change of bias values

        # update layer weights and bias
        layer.weights += change_w
        layer.bias += change_b

    def update_loop(self):
        self.loop_number += 1  # update loop number
