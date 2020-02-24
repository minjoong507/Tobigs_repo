from random import seed
from random import random
import numpy as np
import math


# 네트워크 초기 설정
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


seed(1)
network = initialize_network(2, 1, 2)

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def sigmoid(activation):
    return 1/(1+math.exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

row = [1, 0, None]
output = forward_propagate(network, row)
for layer in network:
    for i in layer:
        print(i['output'])
print("바뀌고-------------")

def sigmoid_derivative(output):
    result = sigmoid(output) * (1 - sigmoid(output))
    return result

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output']) # 역전파시 오차는 어떻게 설정했나요?
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output']) # 시그모이드 함수를 사용한 역전파


expected = [0, 3, 2]
print(expected.index(3))

# backward_propagate_error(network, expected)
# for layer in network:
#     for i in layer:
#         print(i['output'])
#
#         sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
# for i in range(len(expected)):
#     sum_error += (expected[i] - outputs[i]) ** 2