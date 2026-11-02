import numpy as np
from pathlib import Path

class NeuralNet:
    def relu(self, z):
        return np.maximum(0.01 * z, z)
        
    def d_relu(self, z):
        return np.where(z > 0, 1, 0.01)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def d_sigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
        
    def forward(self, x, w1, b1, w2, b2):
        z1 = np.dot(w1, x) + b1
        h1 = self.relu(z1)
        
        z2 = np.dot(w2, h1) + b2
        h2 = self.sigmoid(z2)   # output layer
        
        return z1, h1, z2, h2
        
    def training(self, x, w1, b1, w2, b2, t, lr, eps):
        for ep in range(eps):
            z1, h1, z2, h2 = self.forward(x, w1, b1, w2, b2)           
            # loss
            l = np.mean((h2 - t) ** 2) / h2.size
            
            # output layer
            dl_dh2 = 2 * (h2 - t) / h2.size
            dh2_dz2 = self.d_sigmoid(z2)
            dl_dz2 = dl_dh2 * dh2_dz2
            dl_dw2 = np.dot(dl_dz2, h1.T)
            dl_db2 = np.sum(dl_dz2, axis=1, keepdims=True)

            # hidden layer
            dl_dh1 = np.dot(w2.T, dl_dz2)
            dh1_dz1 = self.d_relu(z1)
            dl_dz1 = dl_dh1 * dh1_dz1
            dl_dw1 = np.dot(dl_dz1, x.T)
            dl_db1 = np.sum(dl_dz1, axis=1, keepdims=True)    

            # update
            w1 -= lr * dl_dw1
            b1 -= lr * dl_db1
            w2 -= lr * dl_dw2
            b2 -= lr * dl_db2
            
            if ep % 10000 == 0:
                print(f"ep:{ep}, loss:{l:.8f}")
        
        print("\nFinal Output:")
        print(np.round(h2, 1))
        return w1, b1, w2, b2
    
# XOR DATASET
x = np.array([
    [0, 0, 1, 1],   # features 1
    [0, 1, 0, 1]    # features 2
])  # shape: (2, 4)

t = np.array([[0, 1, 1, 0]])  # (1, 4)
                              # x = (2, 4)
w1 = np.random.randn(4, 2) * np.sqrt(2 / 2)   # 4 neurons hidden
b1 = np.zeros((4, 1))   # h1 = (4, 4)

w2 = np.random.randn(1, 4) * np.sqrt(2 / 4)   # 1 output
b2 = np.zeros((1, 1))   # h2 = (1, 4)

train = 0
save = 0

location = Path("Neural Networks/Saved Items") / "xor_weights.npz"
#location.parent.mkdir(parents=True, exist_ok=True)

net = NeuralNet()

if train and save:
    w1, b1, w2, b2 = net.training(x, w1, b1, w2, b2, t, lr = 0.15, eps = 100000)
    np.savez(location, w1 = w1, b1 = b1, w2 = w2, b2 = b2)
    print(f"File saved to : {location}")

elif train:
    w1, b1, w2, b2 = net.training(x, w1, b1, w2, b2, t, lr = 0.15, eps = 100000)

else:
    print(f"File opened from : {location}")
    data = np.load(location)
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']

    X = np.array([
    [0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1]])

    _,_,_, h2 = net.forward(X, w1, b1, w2, b2)

    print(np.round(h2, 1))



"""
print(f"w1={w1}\nb1={b1}\nw2={w2}\nb2={b2}")
w1 = np.array([
    [ 1.99995642, -1.52765742],
    [ 2.11122995, -2.11121191],    trained weights and biases
    [ 1.13100744,  1.13100808],
    [-2.68550725,  2.68553007]])
b1 = np.array([
    [-5.69989994e-06],
    [-1.81742563e-05],
    [-1.13100998e+00],
    [-9.34757764e-07]])
w2 = np.array([
    [ 2.05357124,  3.04911646, -2.07901636,  3.92574908]])
b2 = np.array([
    [-4.69415983]])
X = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1]])  
_,_,_, h2 = net.forward(X, w1, b1, w2, b2)
print(np.round(h2, 1))
"""