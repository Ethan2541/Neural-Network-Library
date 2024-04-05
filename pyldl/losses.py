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
        lower_bound = -100
        return np.mean(-y*np.max(lower_bound, np.log(yhat)) - (1-y)*np.max(lower_bound, np.log(1-yhat)))
    
    def backward(self, y, yhat):
        lower_bound = 1e-10
        return -y/np.maximum(yhat, lower_bound) + (1-y)/np.maximum(1-yhat, lower_bound)