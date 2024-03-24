from typing import Any
from pyldl.utils import create_minibatches

class Optim(object):
    def __init__(self, net, loss, gradient_step=1e-3):
        self._network = net
        self._loss = loss
        self._gradient_step = gradient_step

    def __call__(self, *args, **kwds):
        return self.step(*args, **kwds)

    def step(self, batch_x, batch_y):
        self._network.zero_grad()
        batch_yhat = self._network(batch_x)
        delta = self._loss.backward(batch_y, batch_yhat)
        self._network.backward(batch_x, delta)
        self._network.update_parameters(self._gradient_step)


def SGD(network, X, y, loss, batch_size=64, gradient_step=1e-3, n_iter=1000):
    batches = create_minibatches(X, y, batch_size)
    optimizer = Optim(network, loss, gradient_step)
    for _ in range(n_iter):
        for (batch_X, batch_Y) in batches:
            optimizer.step(batch_X, batch_Y)