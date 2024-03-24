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
        return np.exp(X) / sum_exp 
    
    def backward_delta(self, input, delta):
        softmax = self(input)
        derivative = softmax * (1 - softmax)
        return delta * derivative

class LogSoftmax(Activation):
    def forward(self, X):
        sum_exp = np.sum(np.exp(X), axis=1, keepdims=True)
        return X - np.log(sum_exp)
    
    def backward_delta(self, input, delta):
        # je sais pas comment faire pour utiliser le forward, mais normalement ce que j'ai c'est bon
        sum_exp = np.log(np.sum(np.exp(input), axis=1, keepdims=True))
        return delta * (1 - np.exp(input) / sum_exp)
        # ???
        #softmax = np.exp(self.forward(input))
        #return delta - softmax
    
    

def CrossEntropy(Activation):
    def forward(self, y, y_hat):
        ll = -np.log(y_hat, y)
        
