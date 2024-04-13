from pyldl.base import Module
from pyldl.exceptions import DimensionMismatchError
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1. / np.sqrt(in_features)
        self._parameters = np.random.uniform(low=-bound, high=bound, size=(in_features, out_features))
        if bias:
            self._bias = np.random.uniform(low=-bound, high=bound, size=(1, out_features))
        self.zero_grad()
        
    def forward(self, X):
        if X.shape[1] != self._parameters.shape[0]:
            raise DimensionMismatchError(f"(_, {self._parameters.shape[0]})", X.shape)
        if self._bias is not None:
            return X @ self._parameters + self._bias
        else:
            return X @ self._parameters
    
    def backward_update_gradient(self, input, delta):
        if input.shape[0] != delta.shape[0]:
            raise DimensionMismatchError(f"(_, {delta.shape[0]})", input.shape)
        self._gradient += input.T @ delta
        if self._bias is not None:
            self._gradient_bias += delta.sum(axis=0)

    def backward_delta(self, input, delta):
        if delta.shape[1] != self._parameters.shape[1]:
            raise DimensionMismatchError(f"(_, {self._parameters.shape[1]})", delta.shape)
        return delta @ self._parameters.T


class Flatten(Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)
    
    def backward_update_gradient(self, input, delta):
        pass
    
    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)
    

class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride, bias=True):
        super().__init__()
        self._stride = stride
        bound = 1. / np.sqrt(chan_in)
        self._parameters = np.random.uniform(low=-bound, high=bound, size=(k_size, chan_in, chan_out))
        if bias:
            self._bias = np.random.uniform(low=-bound, high=bound, size=(1, chan_out))
        self.zero_grad()

    def forward(self, X):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass