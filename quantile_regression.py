import functools

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from pylab import *

ion()

tf.keras.backend.set_floatx("float64")


def set_seed(seed=1):
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    import tensorflow as tf

    if hasattr(tf, "reset_default_graph"):
        tf.reset_default_graph()
    if hasattr(tf.random, "set_random_seed"):
        tf.random.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


def get_data(seed=1):
    set_seed(seed)
    m = 250  # samples
    n_x = 1  # dim of x
    n_tau = 11
    x = (2 * np.random.rand(m, n_x).astype(np.float64) - 1) * 2
    i = np.argsort(x[:, 0])
    x = x[i]  # to make plotting nicer
    sigma = 0.4 * (1 + 5 / (10 * x[:, [0]] ** 2 + 1))
    mu = x ** 2 + 0.3 * x
    z = np.random.randn(m, 1).astype(np.float64)
    y = mu + sigma * z
    # A = np.random.randn(n_x, 1)
    # y = y.dot(A)  # y is 1d
    tau = np.linspace(1.0 / n_tau, 1 - 1.0 / n_tau, n_tau).astype(np.float64)
    tau = tau[:, None]
    return locals()


def make_layers(*, dims, activation="tanh", kernel_constraint="nonneg", kernel_initializer="uniform"):
    if kernel_initializer == "uniform":
        kernel_initializer = keras.initializers.RandomUniform(minval=0, maxval=1)
    if kernel_constraint == "nonneg":
        kernel_constraint = keras.constraints.NonNeg()
    layers = list()
    for i, dim in enumerate(dims):
        if i == len(dims) - 1:
            activation = None
        layers.append(tf.keras.layers.Dense(dim, kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint, activation=activation, dtype=tf.float64,))
    return layers


def reduce_layers(input, layers):
    return functools.reduce(lambda x, y: y(x), [input] + layers)


def logit(x):
    check = tf.reduce_min(x)
    tf.debugging.assert_greater(check, tf.cast(0.0, tf.float64), message=f"logit got {check} < 0")
    tf.debugging.assert_less(check, tf.cast(1.0, tf.float64), message=f"logit got {check} > 1")
    return tf.math.log(x) - tf.math.log(1 - x)


def rho_quantile_loss(tau_y, u):
    tau, y = tau_y
    tf.debugging.assert_rank(y, 2, f"y should be rank 2")
    u = y[:, None, :] - u[None, :, :]
    # tf.debugging.assert_rank(y, 3, f'y should be rank 3')
    tf.debugging.assert_rank(tau, 2, f"tau should be rank 2")
    tau = tau[None, :, :]
    res = u * (tau - tf.where(u <= np.float64(0.0), np.float64(1.0), np.float64(0.0)))
    return tf.reduce_sum(tf.reduce_mean(res, axis=[1, 2]), axis=0)


def rho_expectile_loss(tau_y, u):
    tau, y = tau_y
    tf.debugging.assert_rank(y, 2, f"y should be rank 2")
    u = y[:, None, :] - u[None, :, :]
    # tf.debugging.assert_rank(y, 3, f'y should be rank 3')
    tf.debugging.assert_rank(tau, 2, f"tau should be rank 2")
    tau = tau[None, :, :]
    res = u ** 2 * (tau - tf.where(u <= 0.0, 1.0, 0.0))
    return tf.reduce_sum(tf.reduce_mean(res, axis=[1, 2]), axis=0)


class QuantileNetworkNoX(tf.keras.models.Model):
    """Deep quantile regression. Recall that quantile is defined as the arg min of

        q(tau) = argmin_u E(rho(tau, Y - u)

    where rho(tau, y) = y * (tau - (y < 0))"""

    def __init__(self, *, dims):
        super().__init__()
        self._my_layers = make_layers(dims=dims, activation="tanh")

    def quantile(self, tau):
        tf.debugging.assert_rank(tau, 2, message=f"tau should be rank two for now")
        u = logit(tau)  # map from (0, 1) to (-infty, infty)
        return reduce_layers(u, self._my_layers)

    def call(self, inputs):
        """Use this signature to support keras compile method"""
        tau, y = inputs
        return self.quantile(tau)


def sanity_plot_nox(steps=1000):
    l = get_data()
    tau = l["tau"]
    y = l["y"]
    model = QuantileNetworkNoX(dims=[16, 16, 1])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def one_step():
        with tf.GradientTape() as tape:
            loss = rho_quantile_loss((tau, y), model((tau, y)))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    # model.compile(loss=rho_quantile_loss, optimizer=opt)
    fig = figure(1)
    fig.clf()
    ax = fig.subplots(1, 1)
    n = len(y)
    p = np.linspace(1.0 / n, 1 - 1.0 / n, n)
    i = y[:, 0].argsort()
    ax.plot(p, y[i, 0], ".", label="data", alpha=0.5)
    loss = list()
    for i in range(steps):
        loss.append(one_step())
    q = model.quantile(tau).numpy().squeeze()
    ax.plot(tau, q, "g.-", label='fit', linewidth=2)
    ax.legend()
    ax.set_xlabel("tau: $P(Y < y)$")
    ax.set_ylabel('y')
    ax.set_title('quantile')
    fig.tight_layout()
    fig.show()
    return locals()


class QuantileNetwork(tf.keras.models.Model):
    """Deep quantile regression. Recall that quantile is defined as the arg min of

        q(tau) = argmin_u E(rho(tau, Y - u)

    where rho(tau, y) = y * (tau - (y < 0))"""

    def __init__(self, *, tau_dims, x_dims, final_dims):
        super().__init__()
        self._my_tau_layers = make_layers(dims=tau_dims, activation="tanh")
        self._my_x_layers = make_layers(dims=tau_dims, activation="tanh", kernel_constraint=None, kernel_initializer="glorot_uniform")
        self._my_x_layers.append(lambda x: tf.square(x))
        self._final_layers = make_layers(dims=final_dims, activation="linear")

    def quantile(self, tau, x):
        tf.debugging.assert_rank(tau, 2, message=f"tau should be rank two for now")
        u = logit(tau)  # map from (0, 1) to (-infty, infty)
        u = reduce_layers(u, self._my_tau_layers)
        v = reduce_layers(x, self._my_x_layers)
        q = v[:, None, :] * u[None, :, :]
        # this is a sum of monotonic functions with positive coef
        q = reduce_layers(q, self._final_layers)
        return q

    def call(self, inputs):
        """Use this signature to support keras compile method"""
        tau, y, x = inputs
        return self.quantile(tau, x)


def sanity_plot(steps=1000):
    l = get_data()
    tau = l["tau"]
    y = l["y"]
    x = l["x"]
    sigma = l["sigma"]
    mu = l["mu"]
    model = QuantileNetwork(tau_dims=[64, 64], x_dims=[64, 64], final_dims=[1])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def one_step():
        with tf.GradientTape() as tape:
            loss = rho_quantile_loss((tau, y), model((tau, y, x)))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    # does not work with, keras mangles dimensions
    # model.compile(loss=rho_quantile_loss, optimizer=opt)

    fig = figure(1, figsize=(12, 6))
    fig.clf()
    ax = fig.subplots(1, 2)
    ax[0].plot(x[:, 0], y.squeeze(), ".", alpha=0.5, label='data')
    ax[0].plot(x[:, 0], mu, label='mu')
    ax[0].plot(x[:, 0], sigma, label='sigma')
    ax[0].legend()
    ax[0].set_ylabel("y")
    ax[0].set_xlabel(f"x[:,0] (x.shape={x.shape})")
    ax[0].set_title('generating process')
    ax[1].plot(x[:, 0], y.squeeze(), ".", alpha=0.5)
    loss = list()
    for i in range(steps):
        loss.append(one_step())
    q = model.quantile(tau, x).numpy().squeeze()
    ax[1].plot(x[:, 0], q, alpha=0.5)
    ax[1].set_xlabel(f"x[:,0] (x.shape={x.shape})")
    ax[1].set_title('inferred quantiles')
    fig.tight_layout()
    fig.show()
    return locals()

if __name__ == '__main__':
    ioff()
    sanity_plot()
    savefig('q.png')
    sanity_plot_nox()
    savefig('q_nox.png')
