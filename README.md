# PyLDL

Python Lightweight Deep Learning is a project that aims to create a dedicated library to implement basic neural networks in Python. It is an assignement part of the course [Machine Learning](https://dac.lip6.fr/master/ml/) from the first year of Sorbonne Université's computer science master.


## Linear Modules

Linear modules are the most basic elements of any neural network. They are thoroughly tested in the notebook `experiments/linear_network`.

### Linear Module

The `Linear` module implements a basic linear layer, whose forward pass is a dot product with its parameters (or weights) and the input. It is one of the most fundamental component of any neural network in `PyLDL`. It is usually followed by an activation module, and often encapsulated in `Sequential`

### Mean Square Error loss function

The `MSELoss` simply implements the mean square error loss between the predicted and actual outputs. It is the mean of the squared euclidean norm of the difference of both outputs along the features axis. Instead of returning a vector of these norms, we chose to only return the mean as it is more convenient to plot the loss across the different epochs of the training phase.


## Nonlinear Modules

Nonlinear modules are basically activation functions which inherit the characteristics of the module `Activation`. As such, they only have a forward and a backward pass. They don't have any parameter to update or to reset.

### Hyperbolic Tangent activation function

First, we considered the hyperbolic tangent:
$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

It is easily implemented thanks to `numpy`'s built-in function `tanh`. Its derivative is given by:
$$tanh'(x) = 1 - tanh^2(x)$$

### Sigmoid activation function

We also implemented the sigmoid function:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Its derivative is:
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### Softmax activation function

In order to represent distributions in the case of multiclass for example, we implemented a `Softmax` module. The forward pass is given by:
$$softmax(x)_i = \frac{e^{x_i}}{\sum_{k=1}^K e^{x_k}}$$

### Rectified Linear Unit activation function

The ReLU activation function is very useful, especially for convolutional networks. Its expression is simple:
$$relu(x) = \max(0,x)$$


## Encapsulation

### Sequential Network

### Optimization

### A specific type of network: the AutoEncoder