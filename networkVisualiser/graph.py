import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Graph:
    def __init__(self, net, inputWidthRange, inputHeightRange, res):
        self.net = net
        self.inputWidthRange = inputWidthRange
        self.inputWidth = abs(inputWidthRange[1] - inputWidthRange[0])
        self.inputHeightRange = inputHeightRange
        self.inputHeight = abs(inputHeightRange[1] - inputHeightRange[0])
        self.setRes(res)
        self.generateData()
        self.plot = plt.figure()
        self.plot.add_axes()
        self.plot = plt.imshow(
            self.data, cmap='Blues', interpolation='antialiased', origin='lower', extent = 
                [inputWidthRange[0], inputWidthRange[1], inputHeightRange[0], inputHeightRange[1]])
        self.createSliders()
        
    # These sliders exist to help visualise how changes to weights and biases affect the output of the perceptron
    def createSliders(self):
        axS = plt.axes([0.15, 0.9, 0.7, 0.03])
        self.resSlider = Slider(axS, 'resolution', 5, 200, valinit=self.res, valstep=1)
        self.resSlider.on_changed(self.updateRes)

        self.weightSliders = []
        self.biasSliders = []

        y = 0.8
        for (i, layer) in enumerate(self.net.layers):
            for (j, weight) in enumerate(layer.weights):
                for (k, w) in enumerate(weight):
                    s = WeightSlider(i, j, k, y, self)
                    self.weightSliders.append(s)
                    y -= 0.03

        y = 0.8
        for (i, layer) in enumerate(self.net.layers):
            for (j, bias) in enumerate(layer.biases):
                s = BiasSlider(i, j, y, self)
                self.biasSliders.append(s)
                y -= 0.03
                    

    # Slider callbacks to update network properties
    def updateRes(self, res):
        self.setRes(res)
        self.updateGraph()

    def setRes(self, res):
        self.res = int(res)

    # Generate array of classification data using network
    def generateData(self): 
        self.data = np.fromfunction(
            np.vectorize(lambda x, y: self.net.Classify(
                np.array([
                    self.inputWidthRange[0] + self.inputWidth * (x / self.res), 
                    self.inputHeightRange[0] + self.inputHeight * (y / self.res)]))), 
            (self.res, self.res))
        
    def updateGraph(self):
        self.generateData()
        self.plot.set_data(self.data)

    def show(self):
        plt.show(self.plot)

# Wrappers for sliders to allow changing of weights and biases
class BiasSlider:
    def __init__(self, layerIndex, biasIndex, y, graph):
        self.net = graph.net
        self.graph = graph
        self.layerIndex = layerIndex
        self.biasIndex = biasIndex
        b = self.net.layers[layerIndex].biases[biasIndex]
        self.s = Slider(plt.axes([0.1, y, 0.15, 0.03]), f'bias {layerIndex} {biasIndex}', -1, 1, valinit=b)
        self.s.on_changed(self.OnChanged)

    # Slider callback method
    def OnChanged(self, val): 
        self.net.layers[self.layerIndex].biases[self.biasIndex] = val
        self.graph.updateGraph()

class WeightSlider:
    def __init__(self, layerIndex, weightRow, weightCol, y, graph):
        self.net = graph.net
        self.graph = graph
        self.layerIndex = layerIndex
        self.weightRow = weightRow
        self.weightCol = weightCol
        w = self.net.layers[layerIndex].weights[weightRow][weightCol]
        self.s = Slider(plt.axes([0.8, y, 0.15, 0.03]), f'weight {layerIndex} {weightRow}', -1, 1, valinit=w)
        self.s.on_changed(self.OnChanged)

    def OnChanged(self, val): 
        self.net.layers[self.layerIndex].weights[self.weightRow][self.weightCol] = val
        self.graph.updateGraph()