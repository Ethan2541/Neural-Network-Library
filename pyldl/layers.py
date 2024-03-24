import numpy as np

class Sequential(object):
    def __init__(self, *layers):
        self._modules = [layer for layer in layers]
        self._inputs = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, X):
        input = X.copy()
        self._inputs = [input]
        for module in self._modules:
            input = module(input)
            self._inputs.append(input)
        return input

    def backward(self, X, delta):
        for i in reversed(range(len(self._modules))):
            self._modules[i].backward_update_gradient(self._inputs[i], delta)
            delta = self._modules[i].backward_delta(X, delta)

    def update_parameters(self, gradient_step=1e-3):
        for module in self._modules:
            module.update_parameters(gradient_step)