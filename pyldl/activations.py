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
        forward_pass = self(input)
        return delta * (1 - forward_pass) * forward_pass
    

class Softmax(Activation):
    def forward(self, X):
        sum_exp = np.sum(np.exp(X), axis=1, keepdims=True)
        return np.exp(X)/sum_exp 
    
    def backward_delta(self, input, delta):
        softmax = self.forward(input)
        derivative = softmax * (1 - softmax)
        return delta * derivative

