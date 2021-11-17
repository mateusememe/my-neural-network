import numpy as np
import pandas as pd



class NeuralNetwork:
    def __init__(self):
        hiddenLayer = []
        inputs = []
        outputLayer = []
        errorArray = []
        mode = 'linear'
    
    def propag(self, net):
        if self.mode == "linear":
            return net/10
        if self.mode == "logistic":
            return 1 / (1 + np.exp(-net))
        if self.mode == "hyperbolic":
            return (1 - np.exp(-2*net)) / (1 + np.exp(-2*net))
    

    def gradient(self, v):  
        if self.mode == "linear":
            return 0.1
        if self.mode == "logistic":
            return propag(v)*(1-propag(v))
        if self.mode == "hyperbolic":
            return 1 - propag(v)**2
