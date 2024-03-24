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