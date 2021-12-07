import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
from keras.datasets import mnist
import collections
from collections import OrderedDict
import seaborn as sns
from data.load_and_process_data import data
from plot_functions.plts import plot_pca, plot_bar_chart, plot_heatmap, plot_confusion_matrix
from layers.fc_linear import Fully_Connected_Layer
from layers.softmax_crossentropy import Softmax_CrossEntropy_Loss_Layer
from activation_functions.relu import ReLU_Layer
from activation_functions.sigmoid import Sigmoid_Layer
from optimizers.gradient_descent_optimizer import MiniBatch_GradientDescent_Optimizer


def train_and_test(ann):
    # Loading MNIST dataset and splitting data into training and testing sets
    (train_x, train_y), (test_x, test_y) = data()

    # PCA plot
    plt.figure()
    plot_pca(train_x, train_y)
    plt.show()

    # Neural's Network Architecture
    network_architecture = OrderedDict()
    activation_layers = ann.activation_layers
    hidden_layers = ann.hidden_layer_nodes
    network_architecture['input_layer'] = Fully_Connected_Layer(train_x.shape[1], hidden_layers[0])
    for i in range(len(hidden_layers)):
        if activation_layers[i] == "relu":
            network_architecture[f'relu_activation_layer{i + 1}'] = ReLU_Layer()
        if activation_layers[i] == "sigmoid":
            network_architecture[f'sigmoid_activation_layer{i + 1}'] = Sigmoid_Layer()
        if i != len(hidden_layers) - 1:
            network_architecture[f'hidden_layer{i + 1}'] = Fully_Connected_Layer(hidden_layers[i], hidden_layers[i + 1])

    network_architecture['output_layer'] = Fully_Connected_Layer(hidden_layers[len(hidden_layers) - 1], 10)
    network_architecture['activate_loss'] = Softmax_CrossEntropy_Loss_Layer()
    optimizer = MiniBatch_GradientDescent_Optimizer(lr=0.1, lr_decay=5e-7, momentum=0.7)

    # Printing the model's characteristics
    print("-" * 24)
    print("| NETWORK ARCHITECTURE: |")
    print("-" * 24)
    print(f"ANN LAYERS...".ljust(10) + f" {len(hidden_layers) + 2}")
    print(f"HIDDEN LAYERS... ".ljust(10) + f"{len(hidden_layers)}")
    print(f"input layer... input size... {train_x.shape[1]} ...")
    for i in range(len(hidden_layers)):
        print(f"hidden layer{i + 1}... neurons... {hidden_layers[i]}")
        print(f"activation layer{i + 1}... {activation_layers[i]}")
    print("softmax output layer ... neurons... 10")
    print("LOSS... Categorical Cross-Entropy")
    print("OPTIMIZER... Mini-Batch Gradient Descent")

    # Splitting data into batches
    per_batch = 1000
    batches = (len(train_x) // per_batch)
    if batches * per_batch < len(train_x):
        batches += 1

    # Training
    print("\n\n" + "-" * 15)
    print("| TRAINING... |")
    print("-" * 15)
    total_epoch_accuracies = []
    total_epoch_loss = []
    for epoch in range(31):
        total_predictions = []
        network_architecture['activate_loss'].loss.new_epoch()
        batch_accuracies = 0
        batch_accuracies_sum = 0
        for step in range(batches):
            batch_x = train_x[step * per_batch:(step + 1) * per_batch]
            batch_y = train_y[step * per_batch:(step + 1) * per_batch]

            # Forward pass
            previous_layer = None
            for i in network_architecture.keys():
                if i == 'input_layer':
                    network_architecture[i].forward_pass(batch_x)
                if i != 'input_layer' and i != 'activate_loss':
                    network_architecture[i].forward_pass(previous_layer.layer_output)
                if i == 'activate_loss':
                    network_architecture[i].forward_pass(previous_layer.layer_output, batch_y)
                previous_layer = network_architecture[i]

            # Model predictions
            predictions = np.argmax(network_architecture['activate_loss'].layer_output, axis=1)
            total_predictions.append(predictions)

            if len(batch_y.shape) == 2:
                batch_y = np.argmax(batch_y, axis=1)
            classified_correctly = predictions == batch_y
            batch_accuracies_sum = batch_accuracies_sum + np.sum(classified_correctly)
            batch_accuracies = batch_accuracies + len(classified_correctly)
            batch_accuracy = np.mean(classified_correctly)

            # Backward pass
            reversed_keys = reversed(list(network_architecture.keys()))
            previous_layer = None
            for i in reversed_keys:
                if i != 'activate_loss':
                    network_architecture[i].backward_pass(previous_layer.grad_output)
                if i == 'activate_loss':
                    network_architecture[i].backward_pass(network_architecture['activate_loss'].layer_output, batch_y)
                previous_layer = network_architecture[i]

            # Update weights and bias
            optimizer.update_learning_rate()
            for i in network_architecture.keys():
                if i.find('activ') == -1:
                    optimizer.update_weights_bias(network_architecture[i])
            optimizer.update_loop()

        # Printing neural network progress
        epoch_loss = network_architecture['activate_loss'].loss.epoch_loss()
        accuracy = batch_accuracies_sum / batch_accuracies
        total_epoch_accuracies.append(accuracy)
        total_epoch_loss.append(epoch_loss)
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {epoch_loss:.3f}, ' +
              f'lr: {optimizer.lr}')

    # Training Evaluation Plots
    predict = [item for sublist in total_predictions for item in sublist]
    predict_counter = sorted(collections.Counter(predict).items())
    print(collections.Counter(predict))

    labels, values = zip(*predict_counter)
    confusion = confusion_matrix(predict, train_y)

    plt.figure(figsize=(6, 6))
    plot_heatmap(confusion)
    plt.show()

    plt.figure()
    plot_confusion_matrix(confusion, labels, title='')
    plt.show()

    plt.figure()
    plt.plot(total_epoch_accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title("Training Epoch Accuracy")
    plt.show()

    plt.figure()
    plt.plot(total_epoch_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Training Epoch Loss")
    plt.show()

    # Testing
    testing = 30
    imgs = train_x[:testing]
    labels = train_y[:testing]
    previous_layer = None
    for i in network_architecture.keys():
        if i == 'input_layer':
            network_architecture[i].forward_pass(imgs)
        if i != 'input_layer' and i != 'activate_loss':
            network_architecture[i].forward_pass(previous_layer.layer_output)
        if i == 'activate_loss':
            network_architecture[i].forward_pass(previous_layer.layer_output, labels)
        previous_layer = network_architecture[i]
    predictions = np.argmax(network_architecture['activate_loss'].layer_output, axis=1)
    classified_correctly_test = predictions == labels
    test_accuracy = np.mean(classified_correctly)
    print("Testing Accuracy: ", test_accuracy)
    show_img = np.reshape(imgs[0], (28, 28))
    plt.imshow(show_img, cmap='binary')
    plt.title("Example image")
    plt.show()
    print("Predicted Labels: ", predictions)
    print("Actual Labels: ", labels)

