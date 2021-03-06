"""
Some experiment with gibbs sampling as opposed to optimization.
"""
import numpy as np
import pandas as pd
from .quantile_regression import set_seed, get_data

def logit(x):
    return tf.math.log(x) - tf.math.log(1 - x)

def rho_quantile_loss(tau_y, u):
    tau, y = tau_y
    # tf.debugging.assert_rank(y, 2, f"y should be rank 2")
    u = y[:, None, :] - u[None, :, :]
    # tf.debugging.assert_rank(tau, 2, f"tau should be rank 2")
    tau = tau[None, :, :]
    J = u * (tau - np.where(u <= np.float64(0.0), np.float64(1.0), np.float64(0.0)))
    # return final_reduce(J)
    return np.sum(np.mean(J, axis=[1, 2]), axis=0)


def sanity_plot_nox(steps=1000):
    l = get_data()
    tau = l["tau"]
    y = l["y"]
    # model = QuantileNetworkNoX(dims=[16, 16, 1])
    return locals()
