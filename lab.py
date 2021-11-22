from random import random
n_inputs = 6
n_hiddens = 2
n_outputs = 5

hiddenLayer = [{'W': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hiddens)]
outputLayer = [{'W': [random() for _ in range(n_hiddens + 1)]} for _ in range(n_outputs)]

print(hiddenLayer)
print(outputLayer)