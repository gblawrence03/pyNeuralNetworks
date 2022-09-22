# gblawrence03

# Classifies arbitrary data using a single layer perceptron.
# The data is visualised using a graph. 
# The weights and biases of the perceptron can be modified in order
# to visualise the effects the changes have on the classification. 

from network import Network
from graph import Graph

def main():
    inputWidthRange = (-50, 50) # Specify the range of inputs to classify
    inputHeightRange = (-50, 50)
    res = 25 # Resolution of graph
    net = Network([2, 3, 2]) # Layer sizes
    graph = Graph(net, inputWidthRange, inputHeightRange, res) 
    graph.show()
    return

if __name__ == "__main__":
    main()