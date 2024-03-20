from abc import ABC, abstractmethod
import numpy as np

class Module(ABC):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    @abstractmethod
    def zero_grad(self):
        """Sets the gradient of the parameters to zero.
        
        This function is used to reset the gradient of the parameters of the module.
        """
        self._gradient = np.zeros_like(self._parameters)

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
            ndarray: Derivate of the error.
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """Updates the parameters of the module.
        
        This function updates the parameters of the module according to the gradient computed during the backward pass, and the gradient step. It basically performs a gradient descent step.

        Args:
            gradient_step (float): Step size of the gradient descent.
        """
        self._parameters -= gradient_step*self._gradient