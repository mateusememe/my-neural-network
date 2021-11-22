import math
import numpy as np
import pandas as pd
from random import random, randrange
from random import seed

class Network:
    MODE_LINEAR = 0
    MODE_LOGISTIC = 1
    MODE_HYPERBOLIC = 2

    def __init__(self, n_inputs = 1, n_outputs = 1, n_hiddens=None, mode=MODE_LINEAR, learning_rate=0.001, error=0.3,epochs=100):
        seed(1)
        self.error = error
        self.epochs = epochs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.normalization = (0,0) #(min, max)
        self.learning_rate = learning_rate
        self.input = list()
        self.errorArray = list()
        self.dic_encode = dict()
        self.mode = mode

    def propag(self, net):
        if self.mode == self.MODE_LINEAR:
            return net/10
        if self.mode == self.MODE_LOGISTIC:
            return 1 / (1 + np.exp(-net))
        if self.mode == self.MODE_HYPERBOLIC:
            return (1 - np.exp(-2*net)) / (1 + np.exp(-2*net))
    
    def derivative_function(self, v):  
        if self.mode == self.MODE_LINEAR:
            return 0.1
        if self.mode == self.MODE_LOGISTIC:
            return self.propag(v) * (1-self.propag(v))
        if self.mode == self.MODE_HYPERBOLIC:
            return 1 - self.propag(v)**2

    def initialize(self, data):
        if data != None:
            for column in data.drop(data.columns[-1], axis=1):
                data[str(column)] = data[str(column)].astType(float)
            self.input = data.values.tolist()
        
            #class column
            class_list = [row[-1] for row in self.input]
            self.n_outputs = len(set(class_list))
            self.n_inputs = len(self.input)
            
            # Definindo o numero de camadas se nn passado por parâmetro do construtor
            if self.n_hiddens == None:
                self.n_hiddens = math.floor(math.sqrt((float(self.n_inputs * self.n_outputs))))

            self.hiddenLayer = [{'W': [random() for _ in range(self.n_inputs + 1)]} for _ in range(self.n_hiddens)]
            self.outputLayer = [{'W': [random() for _ in range(self.n_hiddens + 1)]} for _ in range(self.n_outputs)]

            # Encoding
            self.dic_encode = dict()
            for i, value in enumerate(set(class_list)):
                self.dic_encode[value] = i
            
            for row in self.input:
                row[-1] = self.dic_encode[row[-1]]
        
            random.shuffle(self.input)
            self.normalize()
        else:
            print("-------------> Erro ao inicilizar a Rede")
    
    def forward_propagate(self, row_data):
        inputs = row_data
        for layer in [self.hiddenLayer, self.outputLayer]:
            next_inputs = list()
            for neuron in layer:
                weights = neuron['W']
                
                # Activation
                net = weights[-1] # initialize with bias
                for i in range(len(weights) - 1):
                    net += weights[i] * inputs[i]
                
                neuron['O'] = self.propag(net)
                next_inputs.append(neuron['O'])
            inputs = next_inputs
                        
        return inputs
    
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):

        #delta calculate for output  layer
        for index in range(len(self.outputLayer)):
            neuron = self.outputLayer[index]
            error = expected[index] - neuron['O']
            neuron['delta'] = error * self.derivative_function(neuron['O'])

        #delta calculate for hidden layer
        for index in range(len(self.hiddenLayer)):
            error = 0.0
            for neuron in self.outputLayer:
                error += (neuron['W'][index] * neuron['delta'])
            neuron = self.hiddenLayer[index]
            neuron['delta'] = error * self.derivative_function(neuron['O'])

    def update_weights(self, data_row):
        inputs = data_row[:-1]
        # Camada oculta
        for neuron in self.hiddenLayer:
            for j in range(len(inputs)):
                neuron['W'][j] += self.learning_rate * neuron['delta'] * inputs[j]
            neuron['W'][-1] += self.learning_rate * neuron['delta']
        
        inputs = [neuron['O'] for neuron in self.hiddenLayer]
        for neuron in self.outputLayer:
            for j in range(len(inputs)):
                neuron['W'][j] += self.learning_rate * neuron['delta'] * inputs[j]
            neuron['W'][-1] += self.learning_rate * neuron['delta']

    def train(self, data = None):
        if data != None:
            self.initialize(data)
            error = 100
            epoch = 0
            while epoch < self.n_epochs and self.error < error:
                sum_error = 0
                for row in self.input:
                    outputs = self.forward_propagate(row)
                    expected = np.zeros(self.n_outputs)
                    expected[row[-1]] = 1
                    sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                    self.backward_propagate_error(expected=expected)
                    self.update_weights(row)
                print('>epoch=%d, error=%.3f' % (epoch, sum_error))
                epoch += 1

            return True
        else:
            print("--------------> Nenhum dado de treino detecado!")
    
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))
    
    def test(self, data):
        # Lista do encode 
        dist = list(self.filter)
        
        self.input = None
        
        self.initialize(data)
        
        pred = list()
        for row in self.input:
            pred.append(self.predict(row))
        fact = [row[-1] for row in self.input]
        return self.accuracy(fact, pred, dist)
    
    def accuracy(self, facts, predicteds, dist):
        res = 0
        for i in range(len(facts)):
            if facts[i] == predicteds[i]:
                res += 1
        acc = res / float(len(facts)) * 100.0

        matrix = [[0 for _ in range(len(dist))] for _ in range(len(dist))]
        for i in range(len(facts)):
            matrix[int(facts[i])][int(predicteds[i])] += 1
        df = pd.DataFrame(data=matrix, columns=list(self.dic_encode), index=list(self.dic_encode))
        return (acc, df)
    
    def normalize(self):
        print("\n\nZIP do self.input => ", zip(*self.input), "\n")
        states = [[[min(col)], max(col)] for col in zip(*self.input)]
        for row in self.input:
            for index in range(len(row) - 1):
                row[index] = (row[index] - states[index][0]) / (states[index][1] - states[index][0])