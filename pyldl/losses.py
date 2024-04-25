from pyldl.base import Loss
from pyldl.exceptions import DimensionMismatchError
import numpy as np

class MSELoss(Loss):
    def forward(self, y, yhat):
        if yhat.shape != y.shape:
            raise DimensionMismatchError(y.shape, yhat.shape)
        return np.mean(np.linalg.norm((y - yhat), axis=1)**2)
    
    def backward(self, y, yhat):
        return -2*(y - yhat)
    

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        exp_yhat = np.exp(yhat - np.max(yhat, axis=-1, keepdims=True))
        return np.mean(-np.sum(y*yhat, axis=1) + np.log(np.sum(exp_yhat, axis=1)))
    
    def backward(self, y, yhat):
        exp_yhat = np.exp(yhat - np.max(yhat, axis=-1, keepdims=True))
        return exp_yhat / np.sum(exp_yhat, axis=-1, keepdims=True) - y
    
class BCELoss(Loss):
    def forward(self, y, yhat):
        epsilon = 1e-12
        y_pred = np.clip(yhat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def backward(self, y, yhat):
        epsilon = 1e-12
        y_pred = np.clip(yhat, epsilon, 1 - epsilon)
        return (y_pred - y) / (y_pred * (1 - y_pred) * y_pred.shape[0])