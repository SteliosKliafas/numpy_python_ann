import numpy as np


class CrossEntropyLoss:
    def forward_pass(self, predicted, labels):
        predicted = np.clip(predicted, 1e-10, 1 - 1e-10)  # probabilities for all classes
        predicted_probabilities = predicted[range(len(predicted)), labels]  # probabilities of classes equal to labels
        log_likelihoods = -np.log(predicted_probabilities)  # cross entropy loss
        data_loss = np.mean(log_likelihoods)  # mean of loss values
        self.batch_losses_sum += np.sum(log_likelihoods)  # batch loss values
        self.batch_losses += len(log_likelihoods)  # number of samples in batch
        return data_loss

    def backward_pass(self, grad_inputs, labels):
        self.grad_output = grad_inputs.copy()  # make safe copy
        self.grad_output[range(len(grad_inputs)), labels] -= 1  # Calculating gradient
        layer_output = self.grad_output / len(grad_inputs)  # Normalizing
        return layer_output

    def epoch_loss(self):
        loss = self.batch_losses_sum / self.batch_losses  # calculate epoch loss
        return loss

    def new_epoch(self):
        self.batch_losses_sum = 0  # reset batch loss variables
        self.batch_losses = 0
