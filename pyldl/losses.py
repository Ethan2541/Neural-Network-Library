from pyldl.base import Loss
from pyldl.exceptions import DimensionMismatchError
import numpy as np

class MSELoss(Loss):
    def forward(self, y, yhat):
        if yhat.shape != y.shape:
            raise DimensionMismatchError(y.shape, yhat.shape)
        return np.linalg.norm((y - yhat), axis=1)**2
    
    def backward(self, y, yhat):
        return -2*(y - yhat)
    

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        exp_yhat = np.exp(yhat - np.max(yhat, axis=-1, keepdims=True))
        return -np.sum(y*yhat, axis=1) + np.log(np.sum(exp_yhat, axis=1))
    
    def backward(self, y, yhat):
        exp_yhat = np.exp(yhat - np.max(yhat, axis=-1, keepdims=True))
        return exp_yhat / np.sum(exp_yhat, axis=1, keepdims=True) - y
    

# class CrossEntropyLoss(Loss):
#     def forward(self, y, yhat):
#         return -(y * np.log(yhat)).sum(axis=1, keepdims=True)
    
#     def backward(self, y, yhat):
#         return yhat.sum(axis=1, keepdims=True) - y

    
# class CrossEntropyLogSoftmax(Loss):
#     def forward (self, y, yhat):
#         return np.log(np.sum(np.exp(yhat), axis=1)) - np.sum(y * yhat, axis=1)

#     def backward(self, y, yhat):
#         return np.exp(yhat) / np.exp(yhat).sum(axis=1) - y