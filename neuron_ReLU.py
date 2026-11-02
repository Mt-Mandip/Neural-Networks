import math

# ReLU activation
def relu(z):
    return max(0, z)

# ReLU derivative
def relu_derivative(z):
    if z > 0:
        return 1
    else:
        return 0
    
#prediction and weighted sum
def single_neuron(input, weight, bias):
    w_sum = input * weight + bias
    pre_output = relu(w_sum)
    return pre_output, w_sum

# training function
def treaning(input, req_output, weight, bias, lr, episodes):
    for ep in range(episodes):
        # forward pass
        pre_output, w_sum = single_neuron(input, weight, bias)
        # loss (mean sq.root eqn)
        loss = (pre_output - req_output) ** 2
        # backpropagation (chain rule of derevatives)
        dloss_dpre_output = 2 * (pre_output - req_output)

        dpre_output_dw_sum = relu_derivative(w_sum)  

        dloss_dw_sum = dloss_dpre_output * dpre_output_dw_sum

        dloss_dweight = dloss_dw_sum * input

        dloss_dbias = dloss_dw_sum
        # update
        weight -= lr * dloss_dweight
        bias -= lr * dloss_dbias
        if ep % 100 == 0:
            print(f"Ep: {ep}, Loss: {loss:.6f}, Prediction: {pre_output:.12f}")

    print("-----Final-----")
    print("Input:", input)
    print("Target:", req_output)
    print("Prediction:", pre_output)
    return weight, bias

#contorls (variables)
input = 0.5   # input >= 0, and less than target
req_output = .012399 # req_output >= 0
weight = 1    # better if not negative
bias = 1   # bias >= 0
episodes = 1000    # treaning sessions

lr = 0.01   # must be positive, but lower is better

w, b = treaning(input, req_output, weight, bias, lr, episodes)
print("-----After Training-----")
print(f"weight: {w}, bias: {b}")