from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, y, yhat):
        """Forward pass of the loss.
        
        This function computes the loss given the target y and the prediction yhat.

        Args:
            y (ndarray): Target data.
            yhat (ndarray): Prediction data.

        Returns:
            float: Loss value.
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