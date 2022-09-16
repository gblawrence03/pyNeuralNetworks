# Single layer perceptron with two inputs and two outputs
import math

class Network:
    def __init__(self, weight_1_1, weight_2_1, weight_1_2, weight_2_2, bias_1, bias_2):
        self.weight_1_1 = weight_1_1 # Weight from first input to first output
        self.weight_2_1 = weight_2_1 # Weight from second input to first output
        self.weight_1_2 = weight_1_2 # etc
        self.weight_2_2 = weight_2_2
        self.bias_1 = bias_1
        self.bias_2 = bias_2

    # Classifies an input.
    # Returns 0 if the first output is activated higher, otherwise 1
    def Classify(self, input_1, input_2, f = None):
        output_1 = input_1 * self.weight_1_1 + input_2 * self.weight_2_1 + self.bias_1
        output_2 = input_1 * self.weight_1_2 + input_2 * self.weight_2_2 + self.bias_2            

        if (f == None):
            if output_1 > output_2: return 0
            else: return 1

        return (f(output_1) - f(output_2))

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

        
