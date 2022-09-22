# Single layer perceptron with two inputs and two outputs
import math
from layer import Layer

class Network:
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.layers = []
        for i in range(len(layerSizes) - 1):
            self.layers.append(Layer(layerSizes[i], layerSizes[i + 1]))

    # Run inputs through network
    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs

    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return outputs.argmax()



        
