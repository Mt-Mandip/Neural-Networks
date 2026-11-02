import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, features, h_layers, h_neurons, o_neurons, train):
        if train:
            self.sizes = [features] + [h_neurons] * h_layers + [o_neurons]
            self.weights = []
            self.biases = []
            for i in range(len(self.sizes) - 1):
                self.weights.append(np.random.randn(self.sizes[i + 1], self.sizes[i]) * np.sqrt(2 / self.sizes[i]))
                self.biases.append(np.zeros((self.sizes[i + 1], 1)))
            self.dl_dw = [np.zeros_like(w) for w in self.weights]
            self.dl_db = [np.zeros_like(b) for b in self.biases]
        else:
            trained_location = Path("Neural Networks/Saved Items") / "digit_mnist_trained.npz"
            data = np.load(trained_location, allow_pickle=True)

            self.weights = data["w"]
            self.biases = data["b"]

            self.weights = list(self.weights)
            self.biases = list(self.biases)
        
    def relu(self, z):
        return np.maximum(0, z)
    
    def d_relu(self, z):
        return np.where(z > 0, 1, 0)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward_pass(self, input):
        Z = []
        H = [input]
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], H[i]) + self.biases[i]
            Z.append(z)

            if i != len(self.weights) - 1:
                h = self.relu(z)
            else:
                h = self.softmax(z)
            H.append(h)

        return Z, H
    
    def training(self, input, target, lr, eps, batch_size):
        num_loops = len(self.weights)
        num_samples = input.shape[1]

        for ep in range(eps):

            # shuffling the batch order of dataset
            perm = np.random.permutation(num_samples)
            input_shuffled = input[:, perm]
            target_shuffled = target[:, perm]

            for i in range(0, num_samples, batch_size):
                curr_input = input_shuffled[:, i: i + batch_size]
                curr_target = target_shuffled[:, i: i + batch_size]
                curr_batch_size = curr_input.shape[1]

                Z, H = self.forward_pass(curr_input)

                # cross-entropy loss
                l = -np.mean(np.sum(curr_target * np.log(H[-1] + 1e-8), axis=0))

                # output/ last layer backpropagation
                dl_dh = (H[-1] - curr_target) / curr_batch_size
                dh_dz = 1
                dl_dz = dl_dh * dh_dz
                self.dl_dw[-1] = np.dot(dl_dz, H[-2].T)
                self.dl_db[-1] = np.sum(dl_dz, axis=1, keepdims=True)

                # for next layers
                pre_error = dl_dz
                pre_weight = self.weights[-1]

                # other layers backpropagation
                for j in range(num_loops -2, -1, -1):
                    dl_dh_curr = np.dot(pre_weight.T, pre_error)
                    dh_dz_curr = self.d_relu(Z[j])
                    dl_dz_curr = dl_dh_curr * dh_dz_curr
                    self.dl_dw[j] = np.dot(dl_dz_curr, H[j].T)
                    self.dl_db[j] = np.sum(dl_dz_curr, axis=1, keepdims=True)

                    # for next loop
                    pre_error = dl_dz_curr
                    pre_weight = self.weights[j]
                
                # updates
                for k in range(num_loops):
                    self.weights[k] -= lr * self.dl_dw[k]
                    self.biases[k] -= lr * self.dl_db[k]

            # loss calc. after each ep
            _, H_1ep_after = self.forward_pass(input)
            l_1ep_after = -np.mean(np.sum(target * np.log(H_1ep_after[-1] + 1e-8), axis=0))

            # seeing progress, and losses
            self.prediction(ep, eps, target, l_1ep_after, H_1ep_after)

        return self.weights, self.biases

    def prediction(self, ep, eps, target, loss, H):
        if ep % 100 == 0:
            predic = np.argmax(H[-1], axis=0)   
            label = np.argmax(target, axis=0)
            acc = np.mean(predic == label)
            print(f"ep: {ep}, Loss: {loss:.6f}, Acc.: {acc:.4f}")

        elif ep == eps - 1:    
            predic = np.argmax(H[-1], axis=0)   
            label = np.argmax(target, axis=0)
            acc = np.mean(predic == label)
            print("______Training Complete_____\n")
            print("________Final Result_______\n")
            print(f"Loss: {loss:.6f}, Acc.: {acc:.4f}")

    def testing_single(self, input, target, number):
          one_input = input[:, number: number + 1]
          one_target = target[:, number: number + 1]
          _, H = self.forward_pass(one_input)
          info = H[-1]        
          predic = np.argmax(info)   
          confidence = np.max(info)
          label = np.argmax(one_target)
          #print(np.round(H[-1], 2), "\n")
          #print("________Result_______\n")
          print(f"Output: {predic}, Confidence: {confidence:.4f}, Truth: {label}\n")

    def testing_all(self, input, target):
        _, H = self.forward_pass(input)
        predic = np.argmax(H[-1], axis=0)
        label = np.argmax(target, axis=0)
        acc = np.mean(predic == label)
        print("________Final Accuracy_______\n")
        print(f"Final Acc.: {acc:.4f}")

    def visualizing(self, input, target, number):
        one_input = input[:, number: number + 1]
        one_target = target[:, number: number + 1]
        _, H = self.forward_pass(one_input)
        info = H[-1]        
        predic = np.argmax(info)   
        confidence = np.max(info)
        label = np.argmax(one_target)
        image = one_input.reshape(28, 28)  # reshaping to 28x28 px image
        plt.imshow(image, cmap='grey')
        plt.title(f"Pred.: {predic}, Confi.: {confidence:.2f}, Truth: {label}")
        plt.axis('off')
        plt.show()


#########################################################################################


# dataset and saved values location
location_mnist = Path("Neural Networks/Saved Items") / "mnist.npz"
trained_location = Path("Neural Networks/Saved Items") / "digit_mnist_trained.npz"

# loading the dataset
data = np.load(location_mnist)
training_input = data["x_train"]   # mnist dataset, train (60000, 28, 28)
training_target = data["y_train"]   # mnist dataset, test (10000, 28, 28)
testing_input = data["x_test"]
testing_target = data["y_test"]
print(f"\nDataset loaded from: {location_mnist}.")   

# normalization of inputs
training_input = training_input / 255 # values betn. 0 and 1
testing_input = testing_input / 255

# flattning input (60000/ 10000, 28, 28) to (60000/ 10000, 784)
training_input = training_input.reshape(60000, 784).T
testing_input = testing_input.reshape(10000, 784).T

def one_hot(target, num_output=10):
    q = np.zeros((num_output, target.size))    #make label value 1 rest 0,
    q[target, np.arange(target.size)] = 1      # 5 becomes [0,0,0,0,0,1,0,0,0,0,0]
    return q

training_target = one_hot(training_target)
testing_target = one_hot(testing_target)

# initializing of the class
net = NeuralNetwork(features=784, h_layers=6, h_neurons=32, o_neurons=10, train=False)

# training and saving of the data
#w, b = net.training(training_input, training_target, lr=0.05, eps=100, batch_size=64)
#np.savez(trained_location, w=np.array(w, dtype=object), b=np.array(b, dtype=object))

# testing and visualizing the trained data
for i in range(10):
    num = np.random.randint(0, 9999)
    net.testing_single(testing_input, testing_target, number=num)

    #net.testing_all(testing_input, testing_target)

    #net.visualizing(testing_input, testing_target, number=num)

"""
# to visualize all wrong predictions
_, H = net.forward_pass(testing_input)
preds = np.argmax(H[-1], axis=0)
labels = np.argmax(testing_target, axis=0)
wrong_indices = np.where(preds != labels)[0]
for i in wrong_indices[:10]:
    net.visualize(testing_input, testing_target, number=i)  
"""     
