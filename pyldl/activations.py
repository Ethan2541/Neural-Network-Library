from pyldl.base import Activation
import numpy as np

class Tanh(Activation):
    def forward(self, X):
        return np.tanh(X)
    
    def backward_delta(self, input, delta):
        forward_pass = self(input)
        return delta * (1 - forward_pass**2)
    

class Sigmoid(Activation):
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        sigmoid = self(input)
        return delta * (1 - sigmoid) * sigmoid
    

class Softmax(Activation):
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)
    
    def backward_delta(self, input, delta):
        softmax = self(input)
        return delta * softmax * (1 - softmax)
    

class ReLU(Activation):
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward_delta(self, input, delta):
        return delta * (input > 0)