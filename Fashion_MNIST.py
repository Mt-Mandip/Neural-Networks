import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.datasets import fashion_mnist

class NeuralNetwork:
  def __init__(self, features, hidden_neuron, hidden_layers, output_neurons):
    self.sizes = [features] + [hidden_neuron] * hidden_layers + [output_neurons]
    self.weights = []
    self.biases = []
    for i in range(len(self.sizes) - 1):
      self.weights.append(np.random.randn(self.sizes[i + 1], self.sizes[i]) * np.sqrt(2 / self.sizes[i]))
      self.biases.append(np.zeros((self.sizes[i + 1], 1)))
    self.dl_dw = [np.zeros_like(w) for w in self.weights]
    self.dl_db = [np.zeros_like(b) for b in self.biases]

  def relu(self, z):
    return np.maximum(z, 0)

  def d_relu(self, z):
    return np.where(z > 0, 1, 0)

  def softmax(self, z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

  def forward_pass(self, input):
    Z = []
    H = [input]
    for i in range(len(self.sizes) - 1):
      z = np.dot(self.weights[i], H[i]) + self.biases[i]
      Z.append(z)
      if i != len(self.weights) - 1:
        h = self.relu(z)
      else:
        h = self.softmax(z)
      H.append(h)
    return Z, H

  def backward_pass(self, input, target, x, t, lr, eps, batch_size):
    for ep in range(eps):
      rand_order = np.random.permutation(input.shape[1])
      rand_ord_input = input[:, rand_order]
      rand_ord_target = target[:, rand_order]

      for i in range(0, input.shape[1], batch_size):
        new_input = rand_ord_input[:, i: i + batch_size]
        new_target = rand_ord_target[:, i: i + batch_size]
        curr_batch_size = new_input.shape[1]

        Z, H = self.forward_pass(new_input)

        loss = -np.mean(np.sum(new_target * np.log(H[-1] + 1e-8), axis=0))

        dl_dh = (H[-1] - new_target) / curr_batch_size
        dh_dz = 1
        dl_dz = dl_dh * dh_dz
        self.dl_dw[-1] = np.dot(dl_dz, H[-2].T)
        self.dl_db[-1] = np.sum(dl_dz, axis=1, keepdims=True)

        pre_error = dl_dz
        pre_w = self.weights[-1]

        for j in range(len(self.weights) - 2, -1, -1):
          dl_dh_curr = np.dot(pre_w.T, pre_error)
          dh_dz_curr = self.d_relu(Z[j])
          dl_dz_curr = dl_dh_curr * dh_dz_curr
          self.dl_dw[j] = np.dot(dl_dz_curr, H[j].T)
          self.dl_db[j] = np.sum(dl_dz_curr, axis=1, keepdims=True)

          pre_error = dl_dz_curr
          pre_w = self.weights[j]

          l2_lambda = 0.0001
        for k in range(len(self.weights)):
          self.weights[k] -= lr * (self.dl_dw[k] + l2_lambda * self.weights[k])
          self.biases[k] -= lr * self.dl_db[k]

        #lr = lr * 0.095

      test_input = x[:, :]
      test_target = t[:, :]
      _, test_H = self.forward_pass(test_input)
      test_loss = -np.mean(np.sum(test_target * np.log(test_H[-1] + 1e-8), axis=0))

      self.show_things(test_H, test_loss, eps, ep, test_target)

    return self.weights, self.biases

  def show_things(self, H, loss, eps, ep, target):
    if ep % 10 == 0:
      predic = np.argmax(H[-1], axis=0)
      label = np.argmax(target, axis=0)
      acc = np.mean(predic == label)
      print(f"ep: {ep}, Loss: {loss:.6f}, Acc.: {acc:.4f}")

    if ep == eps - 1:
      predic = np.argmax(H[-1], axis=0)
      label = np.argmax(target, axis=0)
      acc = np.mean(predic == label)
      print("______Training Complete_____")
      print("________Final Result_______\n")
      print(f"Loss: {loss:.6f}, Acc.: {acc:.4f}\n")

  def testing(self, input, target, number):
      fashion_mnist_labels = [
      "T-shirt",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle boot"]
      one_input = input[:, number: number + 1]
      one_target = target[:, number: number + 1]
      _, H = self.forward_pass(one_input)
      info = H[-1]
      predic = np.argmax(info)
      confidence = np.max(info)
      label = np.argmax(one_target)
      #print(np.round(H[-1], 2), "\n")
      #print("________Result_______\n")
      print(f"\nOutput: {fashion_mnist_labels[int(predic)]}, Conf.: {confidence:.2f}, Truth: {fashion_mnist_labels[int(label)]}")

# importing and managing data before any thing happens.

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_input = x_train / 255
test_input = x_test / 255
train_target = y_train
test_target = y_test

train_input = train_input.reshape(60000, 784).T
test_input = test_input.reshape(10000, 784).T

def one_hot(target, num_output=10):
    q = np.zeros((num_output, target.size))
    q[target, np.arange(target.size)] = 1
    return q

train_target = one_hot(train_target)
test_target = one_hot(test_target)

net = NeuralNetwork(features=784, hidden_neuron=256, hidden_layers=4, output_neurons=10)
w, b = net.backward_pass(train_input, train_target, test_input, test_target, lr=0.01, eps=50, batch_size=128)

weight_bytes = sum(arr.nbytes for arr in w)
bias_bytes = sum(arr.nbytes for arr in b)

print(f"Weights: {weight_bytes:,} bytes")
print(f"Biases : {bias_bytes:,} bytes")

#for i in range(4020, 4030, 1):
 #net.testing(test_input, test_target, number = i)

_,w_ans = net.forward_pass(test_input)
ans = np.argmax(w_ans[-1], axis=0)
truth = np.argmax(test_target, axis=0)
wrong_ans = np.where(ans != truth)[0]
for j in range(min(6, len(wrong_ans))):
  idx = wrong_ans[j]
  image = test_input[:, idx].reshape(28, 28)
  plt.imshow(image, cmap="grey")
  plt.axis("off")
  plt.show()
