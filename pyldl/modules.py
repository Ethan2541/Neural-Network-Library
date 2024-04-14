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
        k_size, chan_in = self._parameters.shape[:2]
        batch_size, length = X.shape[:2]

        d_out = (length - k_size)//self._stride + 1
        X_view = np.lib.stride_tricks.sliding_window_view(X, (1, k_size, chan_in))[:, ::self._stride, :]
        X_view = X_view.reshape(batch_size, d_out, chan_in, k_size)

        output = np.einsum("bdik, kio -> bdo", X_view, self._parameters)
        if self._bias is not None:
            output += self._bias
        return output

    def backward_update_gradient(self, input, delta):
        k_size, chan_in = self._parameters.shape[:2]
        batch_size, length = input.shape[:2]

        d_out = (length - k_size)//self._stride + 1
        X_view = np.lib.stride_tricks.sliding_window_view(input, (1, k_size, chan_in))[:, ::self._stride, :]
        X_view = X_view.reshape(batch_size, d_out, chan_in, k_size)

        self._gradient += np.einsum("bdik, bdo -> kio", X_view, self._parameters) / batch_size
        if self._bias is not None:
            self._gradient_bias += delta.sum(axis=(0,1)) / batch_size

    def backward_delta(self, input, delta):
        k_size = self._parameters.shape[0]
        length = input.shape[1]

        d_out = (length - k_size)//self._stride + 1
        convolution = np.einsum("bdo, kio -> kbdi", delta, self._parameters).astype(input.dtype)

        output = np.zeros_like(input)
        for i in range(k_size):
            output[:, i:i+d_out*self._stride:self._stride, :] += convolution[i]
        return output


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        d_out = (length - self._k_size)//self._stride + 1

        X_view = np.lib.stride_tricks.sliding_window_view(X, (1, self._k_size, 1))[:, ::self._stride, :]
        X_view = X_view.reshape(batch_size, d_out, chan_in, self._k_size)
        return np.max(X_view, axis=-1)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self._k_size) // self._stride + 1

        input_view = np.lib.stride_tricks.sliding_window_view(input, (1, self._k_size, 1))[:, ::self._stride, :]
        input_view = input_view.reshape(batch_size, out_length, chan_in, self._k_size)

        idx = np.argmax(input_view, axis=-1)
        batch_idx, d_idx, chan_idx = np.meshgrid(
            range(batch_size),
            range(out_length),
            range(chan_in),
            indexing="ij",
        )
        output = np.zeros_like(input)
        output[batch_idx, d_idx*self._stride + idx, chan_idx] += delta[batch_idx, idx, chan_idx]
        return output


class Flatten(Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)
    
    def backward_update_gradient(self, input, delta):
        pass
    
    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)