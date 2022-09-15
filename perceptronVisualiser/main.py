# gblawrence03

# Classifies arbitrary data using a single layer perceptron.
# The data is visualised using a graph. 
# The weights and biases of the perceptron can be modified in order
# to visualise the effects the changes have on the classification. 

from network import Network
from graph import Graph

def main():
    inputWidth = 10
    inputHeight = 10
    res = 50 
    net = Network(1, 0, 0, 0, 0, 0) # If these are all set to 0, then the graph will not show correctly
    graph = Graph(net, inputWidth, inputHeight, res) 
    graph.show()
    return

if __name__ == "__main__":
    main()