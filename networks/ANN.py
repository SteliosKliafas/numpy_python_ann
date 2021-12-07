class ANN:
    def __init__(self, activation_layers=[], hidden_layer_nodes=[]):
        # Initializing the neural network architecture
        self.activation_layers = activation_layers
        self.hidden_layer_nodes = hidden_layer_nodes
        while True:
            print("-"*59)
            print("| INITIALIZING THE ARTIFICIAL NEURAL NETWORK ARCHITECTURE |")
            print("-" * 59)
            number_of_hidden_layers = input("\nEnter the number of hidden layers: ")
            if number_of_hidden_layers.isdigit():
                number_of_hidden_layers = int(number_of_hidden_layers)
                break
            else:
                print("\nPlease enter an Integer value for the number of hidden layers")
                continue
        print("\nEnter 'relu' for ReLU activation layer or 'sigmoid' for Sigmoid activation layer")
        i = 0
        while i < number_of_hidden_layers:
            layer_activation = input(f"\nEnter the activation function for hidden layer {i}: ")
            number_of_nodes = input(f"\nEnter the number of neurons for hidden layer {i}: ")
            if number_of_nodes.isdigit() and layer_activation == "relu" or layer_activation == "sigmoid":
                self.activation_layers.append(layer_activation)
                self.hidden_layer_nodes.append(int(number_of_nodes))
                i = i + 1
            else:
                print("\n\nPlease check your inputs and try again")
                print("Note: enter 'relu' or 'sigmoid' for activation layer functions\
                 AND an Integer value for the number of layer neurons")