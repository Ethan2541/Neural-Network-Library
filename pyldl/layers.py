class Sequential(object):
    def __init__(self, *modules):
        self._modules = [module for module in modules]
        self._inputs = None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, X):
        input = X.copy()
        self._inputs = []
        for module in self._modules:
            self._inputs.append(input)
            input = module(input)
        return input

    def backward(self, input, delta):
        for i in reversed(range(len(self._modules))):
            self._modules[i].backward_update_gradient(self._inputs[i], delta)
            delta = self._modules[i].backward_delta(self._inputs[i], delta)

    def update_parameters(self, gradient_step=1e-3):
        for module in self._modules:
            module.update_parameters(gradient_step)

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()