"""
Some experiment with Gibbs sampling as opposed to optimization.

Try to use pure numpy where possible without re-writing the function from the other sections.
"""
import numpy as np
import pandas as pd
import functools
from scipy.special import logit
from .quantile_regression import get_data

class Dense():
    """
    The point is to make something where we can broadcast across weights.
    """
    def __init__(self,
            activation=None,
            kernel_initializer=None,
            kernel_constraint=None,
            dtype=np.float64):
        self._kernel = None
        self._bias = None
        self.kernel_mapping = None
        self.kernel_initializer = kernel_initializer
        self.dtype = dtype
        if kernel_constraint == 'nonneg':
            self.kernel_mapping = np.exp
        self.activation = None
        if self.activation == 'tanh':
            self.activation = np.tanh

    def initialize_weights(self, input_dim, weight_dim, output_dim):
        ki = np.random.randn
        if self.kernel_initializer == 'uniform':
            ki = np.random.rand
        self._kernel = ki(input_dim, weight_dim, output_dim).astype(self.dtype)
        self._bias = np.random.randn(weight_dim, output_dim).astype(self.dtype)

    @property
    def kernel(self):
        return self._kernel if self.kernel_mapping is None else self.kernel_mapping(self._kernel)

    @property
    def bias(self):
        return self._bias

    @property
    def weights(self):
        return [self._kernel, self._bias]

    def set_weights(self, *weights):
        self._kernel = weights[0]
        self._bias = weights[1]

    def __call__(self, inputs):
        # print(f'einsum {inputs.shape} x {self.kernel.shape}')
        x = np.einsum('ijk,jkl->ijl', inputs, self.kernel) #  + self.bias[None, :]
        # print(f'result {x.shape}')
        return x if self.activation is None else self.activation(x)

def make_layers(*, input_dim, weight_dim, dims, activation="tanh", final_activation=None, kernel_constraint="nonneg", kernel_initializer="uniform"):
    """
    A utility for making layers.
    If all kernels are non-negative you should have monotonic property.
    """
    layers = list()
    for i, dim in enumerate(dims):
        if i == len(dims) - 1:
            activation = final_activation
        layer = Dense(kernel_initializer=kernel_initializer,
                      kernel_constraint=kernel_constraint,
                      activation=activation,
                      dtype=np.float64)
        layer.initialize_weights(weight_dim, input_dim, dim)
        layers.append(layer)
        input_dim = dim
    return layers

def reduce_layers(inputs, layers):
    return functools.reduce(lambda x, y: y(x), [inputs] + layers)

class QuantileNetworkNoX():
    def __init__(self, *, input_dim, weight_dim, dims):
        """
        input_dim: feature dim
        weight_dim: number of weight samples to run in parallel
        dims: sequence of output/input dims
        """
        self._my_layers = make_layers(
                input_dim=input_dim,
                weight_dim=weight_dim,
                dims=dims,
                activation="tanh",
                kernel_constraint="nonneg")


    def quantile(self, tau):
        # make tau be 3d because inputs to future layers will be 3d.
        # batch dims are (tau, theta) space
        # inputs are (tau, theta, blah)
        # see the einsum calc for details
        tau = tau[:, None, :]
        u = logit(tau)
        return reduce_layers(u, self._my_layers)

    def __call__(self, inputs):
        tau, y = inputs
        return self.quantile(tau)


def rho_quantile_loss(tau_y, uu):
    tau, y = tau_y
    # we now have 3 batch dims! y x tau x params
    uu = y[:, None, None] - uu[None]
    tau = tau[None, :, None]
    J = uu * (tau - np.where(uu <= 0, 1, 0))
    # NOTE the reduce is different that in the tf example
    # we are preserving differences across the param samples
    # we want to be extrinsic in n_samples, intrinsic in n_tau
    return np.sum(np.mean(J, axis=1).squeeze(), axis=0)


def sanity_plot_nox(steps=1000):
    l = get_data()
    # I can't remember why these were 2d, probably something to do with tf
    tau = l["tau"]
    y = l["y"]
    weight_dim = 20
    model = QuantileNetworkNoX(
            input_dim=1,
            weight_dim=weight_dim,
            dims=[16, 16, 1]
            )
    loss = rho_quantile_loss((tau, y), model((tau, y)))
    return locals()
