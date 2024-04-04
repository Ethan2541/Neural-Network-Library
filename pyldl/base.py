from abc import ABC, abstractmethod
import numpy as np


class Module(ABC):
    def __init__(self):
        self._parameters = None
        self._bias = None
        self._gradient = None
        self._gradient_bias = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def zero_grad(self):
        """Sets the gradient of the parameters to zero.
        
        This function is used to reset the gradient of the parameters of the module.
        """
        if self._parameters is not None:
            self._gradient = np.zeros_like(self._parameters)
        if self._bias is not None:
            self._gradient_bias = np.zeros_like(self._bias)

    @abstractmethod
    def forward(self, X):
        """Forward pass of the module.
        
        This function computes the output of the module given the input X.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Output of the module.
        """
        pass

    @abstractmethod
    def backward_update_gradient(self, input, delta):
        """Updates the gradient of the parameters.

        This function computes the gradient of the parameters of the module given the input and the error of the module.

        Args:
            input (ndarray): Input data.
            delta (ndarray): Derivatives of the network's next layer.
        """
        pass

    @abstractmethod
    def backward_delta(self, input, delta):
        """Computes the derivate of the error.

        This function computes the derivate of the error given the input of the module, and the derivatives of the next layer.
        
        Args:
            input (ndarray): Input data.
            delta (ndarray): Derivatives of the network's next layer.

        Returns:
            ndarray: Derivative of the error with respect to the input and the next layer's delta.
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """Updates the parameters of the module.
        
        This function updates the parameters of the module according to the gradient computed during the backward pass, and the gradient step. It basically performs a gradient descent step.

        Args:
            gradient_step (float): Step size of the gradient descent.
        """
        if self._parameters is not None:
            self._parameters -= gradient_step*self._gradient
        if self._bias is not None:
            self._bias -= gradient_step*self._gradient_bias


class Activation(Module):
    def __init__(self):
        super().__init__()

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def zero_grad(self):
        pass


class Loss(ABC):
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    @abstractmethod
    def forward(self, y, yhat):
        """Forward pass of the loss.
        
        This function computes the loss given the target y and the prediction yhat.

        Args:
            y (ndarray): Target data.
            yhat (ndarray): Prediction data.

        Returns:
            ndarray: Loss value for each observation.
        """
        pass

    @abstractmethod
    def backward(self, y, yhat):
        """Backward pass of the loss.

        This function computes the derivative of the loss given the target y and the prediction yhat.

        Args:
            y (ndarray): Target data.
            yhat (ndarray): Prediction data.

        Returns:
            ndarray: Derivative of the loss.
        """
        pass