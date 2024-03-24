from pyldl.base import Module
from pyldl.exceptions import DimensionMismatchError
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        self._bias = np.random.rand(out_features)
        self._parameters = np.random.rand(in_features, out_features)
        self.zero_grad()
        
    def forward(self, X):
        if X.shape[1] != self._parameters.shape[0]:
            raise DimensionMismatchError(f"(_, {self._parameters.shape[0]})", X.shape)
        return X @ self._parameters + self._bias
    
    def backward_update_gradient(self, input, delta):
        if input.shape[0] != delta.shape[0]:
            raise DimensionMismatchError(f"(_, {delta.shape[0]})", input.shape)
        self._gradient += input.T @ delta
        self._gradient_bias += delta.sum(axis=0)

    def backward_delta(self, input, delta):
        if delta.shape[1] != self._parameters.shape[1]:
            raise DimensionMismatchError(f"(_, {self._parameters.shape[1]})", delta.shape)
        return delta @ self._parameters.T