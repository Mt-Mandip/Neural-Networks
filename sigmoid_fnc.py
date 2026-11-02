"""
#approxmiate of exponational
def exp(w_sum, terms = 50):
    result = 1
    upper = 1
    lower = 1
    for i in range(1, terms):
        upper *= w_sum
        lower *= i
        result += upper / lower
    return result
#activation function Sigmoid
def sigmoid(w_sum):
    return 1/(1 + exp(-w_sum))
#forward pass
def single_neuron(inputs, weights, bias):
    w_sum = 0
    for i in range(len(inputs)):
        w_sum += inputs[i] * weights[i]
    w_sum += bias
    return sigmoid(w_sum)
#changable values
inputs = []
weights = []
quit = False
while not quit:
    value = float(input("Input:"))
    inputs.append(value)
    value2 = float(input("Weight:"))
    weights.append(value2)
    cond = input("Q to quit, Any other to continue: ").capitalize()
    if cond == "Q":
        quit = True
num = float(input("Bias: ")) 
bias = max(0.1, num)
#starting point
output = single_neuron(inputs, weights, bias)
print(f"Output: {output}")

"""
######################################################
######################################################

# single neuron using Sigmoid and ReLU 
import math
import numpy as np
def relu(z):
    return max(0, z)

def sigmoid(z):
    #return 1 /(1 + math.exp(-z))
    return np.maximum(0, z)

def d_relu(z):
    """
    if z > 0:
        return 1
    else:
        return 0
        """
    return (z > 0).astype(float)
    
def single_neuron(i, w, b):
    #z = i * w + b
    z = np.dot(i, w) + b
    po = relu(z)
    #po = sigmoid(z)
    return z, po

def train(i, w, b, t, lr, eps):
    for ep in range(eps):
        z, po = single_neuron(i, w, b)
        l = (po - t) ** 2

        dl_dpo = 2 * (po - t)
    # for ReLU 
        dpo_dz = d_relu(z)
    # for sigmoid
        #dpo_dz = po * (1 - po)
        dl_dz = dl_dpo * dpo_dz
        dl_dw = dl_dz * i
        dl_db = dl_dz

        w -= lr * dl_dw
        b -= lr * dl_db
    print(f"input: {i}, target: {t}, prediction: {po}")
#i = 0.5  # single input
i = np.array([1, .3, .5, 2, .9])   # for 1D input vector
b = 0.1
t = .987123
#w = 1   # single weight for single input
w = np.array([0.1, 0.5, 1, 2, 1.1])   #for 1D weights vector
lr = 0.01
eps = 1000
train(i, w, b, t, lr, eps)
