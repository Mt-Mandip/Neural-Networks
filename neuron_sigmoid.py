import math

#sigmoid activation
def sigmoid(w_sum):
    return 1/(1 + math.exp(-w_sum))

#weighted sum
def single_neuron(input, weight, bias):
    w_sum = input*weight + bias
    pre_output = sigmoid(w_sum)
    return pre_output, w_sum

#treaning function
def treaning(input,req_output,weight,bias,lr,episodes):

    for ep in range(episodes):

        #forward pass
        pre_output, w_sum = single_neuron(input,weight,bias)
        #loss calc. mean sq.root eqn
        loss = (pre_output - req_output) ** 2
        #backpropragation
        dloss_dpre_output = 2 * (pre_output - req_output)

        dpre_output_dw_sum = pre_output*(1 - pre_output)

        dloss_dw_sum = dloss_dpre_output * dpre_output_dw_sum

        dloss_dweight =  dloss_dw_sum * input

        dloss_dbias =  dloss_dw_sum

        #update bias and weight
        weight -= lr * dloss_dweight

        bias -= lr * dloss_dbias

        if ep % 500 == 0:
            print(f"Ep: {ep}, Loss: {loss:.6f}, Prediction: {pre_output:.12f}")
    
    print("-----Final Result-----")
    print("Input:",input)
    print("Target:",req_output)
    print("Prediction:",pre_output)

    return weight, bias

#changable values
input = .01010101    # 0 <= input <= 1
weight = 0.1
bias = 0.1
req_output = .12345678955   # 0 <= req_output <= 1

lr = .2            # 0.5 <= learning rate <= 0.5
episodes = 5500

w, b = treaning(input,req_output,weight,bias,lr,episodes)

print("-----After Traning-----")
print(f"weight: {w}, bias:{b}")