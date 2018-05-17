# neural_network_from_scratch_MNIST

Build a neural network from scratch and train it on MNIST (digit image recognition) using backpropagation.

The variables for the neural network (weights, biases, activations) are stored in the global environment. That also means that functions which use or modify these variables take them from the global environment and store them in the global environment. In R, this should generally be avoided, the better solution would be to store these in lists, but for a script that simple it is acceptable. 

## File structure
    ├── functions.R                     # Function definitions
    ├── neural_network.R                # Load data, initialize network, train, evaluate
    ├── train.csv                       # data
