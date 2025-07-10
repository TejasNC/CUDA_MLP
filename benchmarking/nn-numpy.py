import numpy as np

class Module:
    def update(self, learning_rate):
        pass

class Linear(Module):

    def __init__(self, m, n):
        self.m = m
        self.n = n
        np.random.seed(69)
        self.W = np.random.normal(
            0, np.sqrt(2 / (self.n + self.m)), size=(self.m, self.n)
        )
        self.W0 = np.zeros((self.n, 1))

    def forward(self, A):
        self.A = A
        self.Z = np.dot(np.transpose(self.W), A) + self.W0
        return self.Z

    def backward(self, dLdZ):
        self.dLdW = np.dot(self.A, np.transpose(dLdZ))
        self.dLdW0 = np.mean(dLdZ, axis=1, keepdims=True)
        return np.dot(self.W, dLdZ)

    def update(self, learning_rate):
        self.W -= learning_rate * self.dLdW
        self.W0 -= learning_rate * self.dLdW0

class ReLU(Module):

    def forward(self, Z):
        self.mask = (np.sign(Z) > 0).astype(float)
        return self.mask * Z

    def backward(self, dLdA):
        return self.mask * dLdA

class Tanh(Module):

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        return (1 - (self.A) ** 2) * dLdA

class SoftMax(Module):

    def forward(self, Z):
        clip = np.max(Z, axis=0, keepdims=True)
        return np.exp(Z - clip) / np.sum(np.exp(Z - clip), axis=0, keepdims=True)

    def backward(self, dLdA):
        return dLdA

class CrossEntropyLoss(Module):

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(y_true * np.log(y_pred + 1e-15))  # Added small epsilon for numerical stability

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_pred.shape[1]  # Normalized by batch size

    def predict_class(self, y_pred):
        return np.argmax(y_pred, axis=0)


class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
