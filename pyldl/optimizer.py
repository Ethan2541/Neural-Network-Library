class Optim(object):
    def __init__(self, net, loss, eps):
        self._network = net
        self._loss = loss
        self._eps = eps

    def step(self, batch_x, batch_y):
        batch_yhat = self._network.forward(batch_x)
        delta = self._loss.backward(batch_y, batch_yhat)
        self._network.backward(batch_x, delta)
        self._network.update_parameters(self._eps)