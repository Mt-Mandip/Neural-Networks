# no hidden layer, but multile neurons in single layer
#  cant do much but, tranin on roandom target with random initialized inputs and weights. 
import numpy as np
class Neuron:
    def relu(self, z):
        return np.maximum(0 ,z)
    def d_relu(self, z):
        return (z > 0).astype(float)
    def forward(self, i, w, b):
        z = np.dot(w, i) + b    # size is (neuron, batch)
        po = self.relu(z)   # predicted/ output size is (neuron, features)
        return z, po
    def traning(self, i, w, b, t, lr, eps):
        for ep in range(eps):
            # forward passing of values
            z, po = self.forward(i, w, b)
            # loss calculation
            l = np.mean((po - t) ** 2)
            # backpropagation
            dl_dpo = 2 * (po - t)
            dpo_dz = self.d_relu(z)
            dl_dz = dl_dpo * dpo_dz
            # gradients
            dl_dw = np.dot(dl_dz, i.T)  # size is (neuron, features)
            dl_db = np.sum(dl_dz, axis = 1, keepdims = True)    # size is (neuron, 1)
            # updates
            w -= lr * dl_dw
            b -= lr * dl_db
            if ep % 100 == 0:
                print(f"Ep: {ep}, Loss: {l:.10f}")
        print("-----Final Results-----\n")
        print(f"Input:\n{i}\n")
        print(f"Target:\n{t}\n")
        print(f"Output:\n{po}\n")
        print(f"weights:\n{w}\n")
        print(f"bias:\n{b}\n")
neuron1 = Neuron()
batch = 2      # no. of vectorized inputs
neuron = 4     # no. of neurons             Note: neuron must = features in size
features = 3   # no. of input elements 
#i = np.random.randn(features, batch)
i = np.round(np.random.uniform(-1, 1, (features, batch)), 4)
w = np.random.randn(neuron, features) * 0.01
b = np.ones((neuron, 1)) * 0.1
#t = np.random.randn(neuron, batch)
t = np.round(np.random.uniform(0, 1, (neuron, batch)), 5)
lr = 0.01
eps = 1000
neuron1.traning(i, w, b, t, lr, eps)



