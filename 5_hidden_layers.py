import numpy as np

class NeuralNet:
    def relu(self, z):
        return np.maximum(0.01 * z, z)
    
    def d_relu(self, z):
        return np.where(z > 0, 1, 0.01)
    
    def forward_pass(self, x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
        # layer 1
        z1 = np.dot(w1, x) + b1
        po1 = self.relu(z1)
        # layer 2
        z2 = np.dot(w2, po1) + b2
        po2 = self.relu(z2)
        # layer 3
        z3 = np.dot(w3, po2) + b3
        po3 = self.relu(z3)
        # layer 4
        z4 = np.dot(w4, po3) + b4
        po4 = self.relu(z4)
        # layer 5
        z5 = np.dot(w5, po4) + b5
        po5 = z5

        return z1, po1, z2, po2, z3, po3, z4, po4, z5, po5
    
    def traning(self, x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, t, lr, eps):
        for ep in range(eps):
            # forward pass
            z1, po1, z2, po2, z3, po3, z4, po4, _, po5 = self.forward_pass(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)

            # loss calculation
            l = np.mean((po5 - t) ** 2)

            # backward layer 5's error
            dl_dpo5 = 2 * (po5 - t) / batch # MSE (mean squered error) derevative

            # output layer
            dl_dw5 = np.dot(dl_dpo5, po4.T) # .product of error and input value for layer 5
            dl_db5 = np.sum(dl_dpo5, axis=1, keepdims=True) # bias is independent of input, same bias for whole batch, hence the long code.

            # layer 4
            dl_dpo4 = np.dot(w5.T, dl_dpo5) # to see how much o/p of L4 added to error
            dl_dz4 = dl_dpo4 * self.d_relu(z4) # relu(z) dec or inc error so, so *ing by d_relu(z)
            dl_dw4 = np.dot(dl_dz4, po3.T)
            dl_db4 = np.sum(dl_dz4, axis=1, keepdims=True)  # change in weights and bias

            # layer 3
            dl_dpo3 = np.dot(w4.T, dl_dz4)
            dl_dz3 = dl_dpo3 * self.d_relu(z3)
            dl_dw3 = np.dot(dl_dz3, po2.T)
            dl_db3 = np.sum(dl_dz3, axis=1, keepdims=True)

            # layer 2
            dl_dpo2 = np.dot(w3.T, dl_dz3)
            dl_dz2 = dl_dpo2 * self.d_relu(z2)
            dl_dw2 = np.dot(dl_dz2, po1.T)
            dl_db2 = np.sum(dl_dz2, axis=1, keepdims=True)

            # layer 1
            dl_po1 = np.dot(w2.T, dl_dz2)
            dl_dz1 = dl_po1 * self.d_relu(z1)
            dl_dw1 = np.dot(dl_dz1, x.T) # x being the orginal input
            dl_db1 = np.sum(dl_dz1, axis=1, keepdims=True)

            # updates
            w5 -= lr * dl_dw5
            b5 -= lr * dl_db5

            w4 -= lr * dl_dw4
            b4 -= lr * dl_db4

            w3 -= lr * dl_dw3
            b3 -= lr * dl_db3

            w2 -= lr * dl_dw2
            b2 -= lr * dl_db2

            w1 -= lr * dl_dw1
            b1 -= lr * dl_db1

            if ep % 3000 == 0:
                print(f"Ep: {ep}, Loss: {l:.10f}")

        print("\n-----Final Results-----\n")
        #print(f"Input:\n{x}\n")
        print(f"Target:\n{np.round(t, 3)}\n")
        print(f"Output:\n{np.round(po5, 3)}\n")
    
        #return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5
    
##########################################################################################
##########################################################################################

nn1 = NeuralNet()

features = 2
batch = 100
output_n = 1   # no of answers, no of output neurons
hidden_n = 5    # no of neurons in hidden layers, in this case 5 neuron in each 5 hidden layers

#x = np.round(np.random.uniform(-1, 1, (features, batch)), 5)
#t = np.round(np.random.uniform(-1, 1, (output_n, batch)), 5)

x = np.round(np.random.uniform(-1, 1, (features, batch)), 2)
#t = x[0:1, :] + x[1:2, :]
t = np.sin(5 * x[0:1, :]) + (x[1:2, :] ** 3)

#weights and bias
w1 = np.random.randn(hidden_n, features) * np.sqrt(2 / features)
b1 = np.zeros((hidden_n, 1))

w2 = np.random.randn(hidden_n, hidden_n) * np.sqrt(2 / hidden_n)
b2 = np.zeros((hidden_n, 1))

w3 = np.random.randn(hidden_n, hidden_n) * np.sqrt(2 / hidden_n)
b3 = np.zeros((hidden_n, 1))

w4 = np.random.randn(hidden_n, hidden_n) * np.sqrt(2 / hidden_n)
b4 = np.zeros((hidden_n, 1))

w5 = np.random.randn(output_n, hidden_n) * np.sqrt(2 / hidden_n)
b5 = np.zeros((output_n, 1))

lr = 0.15
eps = 20000

nn1.traning(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, t, lr, eps)











