import numpy as np

class NeuralNetwork:
    def relu(self, z):
        return np.maximum(z * 0.01, z)
    
    def d_relu(self, z):
        return np.where(z > 0, 1, 0.01)
    
    def forward_pass(self, x, W, B):
        Z = []
        H = [x]
        for i in range(len(W)):
            z = np.dot(W[i], H[-1]) + B[i]
            Z.append(z)

            if i == len(W) - 1:
                h = z 
            else:
                h = self.relu(z)
            H.append(h)

        return Z, H
    
    def treaning(self, x, W, B, t, lr, eps):
        # calculate the batch size dynamically from the input matrix
        batch = x.shape[1]
        num_layers = len(W)

        for ep in range(eps):
            Z, H = self.forward_pass(x, W, B)

            # loss calculation for final output
            L = np.mean((H[-1] - t) ** 2)

            dl_dw = [np.zeros_like(w) for w in W]
            dl_db = [np.zeros_like(b) for b in B]

            # output layer error
            dl_dh_last = 2 * (H[-1] - t) / batch  #MSE derivatave
            dl_dw[-1] = np.dot(dl_dh_last, H[-2].T)
            dl_db[-1] = np.sum(dl_dh_last, axis=1, keepdims=True)

            dl_dz_next = dl_dh_last
            w_next = W[-1]

            # Backpropagate through hidden layers (Reverse loop)
            for i in range(num_layers -2, -1, -1):
                dl_dh_curr = np.dot(w_next.T, dl_dz_next)
                dl_dz_curr = dl_dh_curr * self.d_relu(Z[i])

                dl_dw[i] = np.dot(dl_dz_curr, H[i].T)
                dl_db[i] = np.sum(dl_dz_curr, axis=1, keepdims=True)

                # Update next variables for the next iteration of the chain rule
                dl_dz_next = dl_dz_curr
                w_next = W[i]
            
            # final updates for W, B
            for i in range(num_layers):
                W[i] -= lr * dl_dw[i]
                B[i] -= lr * dl_db[i]
            
            if ep % 200 == 0:
                print(f"Ep: {ep}, Loss: {L:.10f}")

        print("\n-----Treaning Complete-----\n")
        print(f"Target:\n{t}\n")
        print(f"Output:\n{np.round(H[-1], 2)}\n")

        return W, B

                    # actual dynamic code(object) above
####################################################################################################
####################################################################################################
                    # testing call of object
NN1 = NeuralNetwork()

features = 2
batch = 100
hidden_neurons = 2
hidden_layers = 1      # total layers = hidden layer + 1
output_neurons = 1

x = np.round(np.random.uniform(-1, 1, (features, batch)), 2)
t = x[0:1, :] + x[1:2, :]  # for +ion, f = 2, batach = 500, hidden_n,_l = 2, 1, output = 1, eps = 2000, lr = 0.05 
#t = np.sin(5 * x[0:1, :]) + (x[1:2, :] ** 3)

# Dynamic Weight & Bias Initialization
# Create an array of our layer sizes to easily generate weights matrices
layer_sizes = [features] + [hidden_neurons] * hidden_layers + [output_neurons]

W = []
B = []

# Loop to create dynamic w1, w2, w3, etc. using He Initialization
for i in range(len(layer_sizes) - 1):
    w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i])
    b = np.zeros((layer_sizes[i+1], 1))
    W.append(w)
    B.append(b)

lr = 0.05
eps = 2000

n_W, n_B = NN1.treaning(x, W, B, t, lr, eps)

"""
X = np.array([
        [0.2, .2, .20, .40, 1, 0, 1, 0.9, -1],             # new input must be betn. -1 and 1
        [0.2, .5, .10, -.30, -1, 1, 1, 0.9, -1]
        ])
"""
quit = True
while not quit:
    a = float(input("Enter a no (-1 < no. < 1): "))
    b = float(input("Enter another no (-1 < no. < 1): "))
    X = np.array([
            [a],
            [b]]) 
    _, HH = NN1.forward_pass(X, n_W, n_B)

    print(np.round(HH[-1], 2))
    cond = input("Q to quit, Any other to continue: ").capitalize()
    if cond == "Q":
        quit = True
        



