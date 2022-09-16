# Single layer perceptron with two inputs and two outputs

class Network:
    def __init__(self, weight_1_1, weight_2_1, weight_1_2, weight_2_2, bias_1, bias_2):
        self.weight_1_1 = weight_1_1 # Weight from first input to first output
        self.weight_2_1 = weight_2_1 # Weight from second input to first output
        self.weight_1_2 = weight_1_2 # etc
        self.weight_2_2 = weight_2_2
        self.bias_1 = bias_1
        self.bias_2 = bias_2

    # Classifies an input. This is done in the simplest possible way with no activation function
    # Returns 0 if the first output is activated higher, otherwise 1
    def Classify(self, input_1, input_2):
        output_1 = input_1 * self.weight_1_1 + input_2 * self.weight_2_1 + self.bias_1
        output_2 = input_1 * self.weight_1_2 + input_2 * self.weight_2_2 + self.bias_2

        if output_1 > output_2: 
            return 0
        else: 
            return 1
