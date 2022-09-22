import numpy as np

class Layer:
    def __init__(self, nodesIn, nodesOut):
        self.nodesIn = nodesIn
        self.nodesOut = nodesOut

        self.weights = np.random.uniform(-1, 1, (nodesOut, nodesIn))
        self.biases = np.random.uniform(-1, 1, nodesOut)

    # Calculate the output values of the layer by multiplying the inputs
    # by the weights and adding the biases
    def CalculateOutputs(self, inputs):
        outputs = np.empty((self.nodesOut))
        for i in range(self.nodesOut):
            outputs[i] = self.ActivationFunction(inputs.dot(self.weights[i]) + self.biases[i])
        return outputs

    def ActivationFunction(self, input):
        return self.sigmoid(input)

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
