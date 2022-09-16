import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Graph:
    def __init__(self, net, inputWidth, inputHeight, res):
        self.net = net
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.setRes(res)
        self.generateData()
        self.plot = plt.figure()
        self.plot.add_axes()
        self.plot = plt.imshow(
            self.data, cmap='Blues', interpolation='nearest', origin='lower', extent = [0, inputWidth, 0, inputHeight])
        self.createSliders()
        
    # These sliders exist to help visualise how changes to weights and biases affect the output of the perceptron
    def createSliders(self):
        axS = plt.axes([0.15, 0.9, 0.7, 0.03])
        self.resSlider = Slider(axS, 'resolution', 5, 200, valinit=self.res, valstep=1)
        self.resSlider.on_changed(self.updateRes)

        axB1 = plt.axes([0.85, 0.8, 0.1, 0.03])
        self.bias_1Slider = Slider(axB1, 'Bias 1', -1, 1, valinit=self.net.bias_1)
        self.bias_1Slider.on_changed(self.updateBias_1)

        axB2 = plt.axes([0.85, 0.7, 0.1, 0.03])
        self.bias_2Slider = Slider(axB2, 'Bias 2', -1, 1, valinit=self.net.bias_2)
        self.bias_2Slider.on_changed(self.updateBias_2)

        axW11 = plt.axes([0.85, 0.5, 0.1, 0.03])
        self.weight_1_1Slider = Slider(axW11, 'Weight 1 1', -1, 1, valinit=self.net.weight_1_1)
        self.weight_1_1Slider.on_changed(self.updateWeight_1_1)

        axW21 = plt.axes([0.85, 0.4, 0.1, 0.03])
        self.weight_2_1Slider = Slider(axW21, 'Weight 2 1', -1, 1, valinit=self.net.weight_2_1)
        self.weight_2_1Slider.on_changed(self.updateWeight_2_1)

        axW12 = plt.axes([0.85, 0.3, 0.1, 0.03])
        self.weight_1_2Slider = Slider(axW12, 'Weight 1 2', -1, 1, valinit=self.net.weight_1_2)
        self.weight_1_2Slider.on_changed(self.updateWeight_1_2)

        axW22 = plt.axes([0.85, 0.2, 0.1, 0.03])
        self.weight_2_2Slider = Slider(axW22, 'Weight 2 2', -1, 1, valinit=self.net.weight_2_2)
        self.weight_2_2Slider.on_changed(self.updateWeight_2_2)

    # Slider callbacks to update network properties
    def updateRes(self, res):
        self.setRes(res)
        self.updateGraph()

    def setRes(self, res):
        self.res = int(res)

    def updateBias_1(self, val):
        self.net.bias_1 = val
        self.updateGraph()

    def updateBias_2(self, val):
        self.net.bias_2 = val
        self.updateGraph()

    def updateWeight_1_1(self, val):
        self.net.weight_1_1 = val
        self.updateGraph()

    def updateWeight_2_1(self, val):
        self.net.weight_2_1 = val
        self.updateGraph()

    def updateWeight_1_2(self, val):
        self.net.weight_1_2 = val
        self.updateGraph()

    def updateWeight_2_2(self, val):
        self.net.weight_2_2 = val
        self.updateGraph()

    # Generate array of classification data using network
    def generateData(self): 
        self.data = np.fromfunction(np.vectorize(
            lambda x, y: self.net.Classify(self.inputWidth * (x / self.res), self.inputHeight * (y / self.res))
        ), (self.res, self.res))
        
    def updateGraph(self):
        self.generateData()
        self.plot.set_data(self.data)

    def show(self):
        plt.show(self.plot)
