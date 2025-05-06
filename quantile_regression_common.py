import random

import matplotlib.pyplot as plt
import numpy as np


def get_data(seed=1, m=250, n_x=1, n_tau=11, L=2):
    """
    x ~ U(-2, 2)
    y ~ N(mu(x), sigma(x))
    """
    random.seed(seed)
    x = (2 * np.random.rand(m, n_x).astype(np.float64) - 1) * 2
    i = np.argsort(x[:, 0])
    x = x[i]  # to make plotting nicer
    sigma = 0.4 * (1 + 5 / (10 * x[:, [0]] ** 2 + 1))
    mu = x**2 + 0.3 * x
    z = np.random.randn(m, 1).astype(np.float64)
    y = mu + sigma * z
    # yc and tau are same dimension, similar functionality
    # this is confusing because mu, sigma are across samples mu(x)
    # want yc for all x here.
    # cheating to know the ranges, but whatever.
    # will be good to see what happens where there is no data.
    yc_max = np.max(y)
    yc_min = np.min(y)
    yc = np.linspace(yc_min, yc_max, n_tau).astype(np.float64)
    yc = yc[:, None]
    # A = np.random.randn(n_x, 1)
    # y = y.dot(A)  # y is 1d
    tau = np.linspace(1.0 / n_tau, 1 - 1.0 / n_tau, n_tau).astype(np.float64)
    tau = tau[:, None]
    return locals()


def set_consistent_figure_params():
    """Set consistent figure parameters for both TF and JAX versions"""
    plt.rcParams.update(
        {
            'figure.figsize': (10, 6),  # Default figure size
            'figure.dpi': 100,  # Figure resolution
            'savefig.dpi': 150,  # Saved figure resolution
            'font.size': 10,  # Default font size
            'axes.titlesize': 12,  # Title font size
            'axes.labelsize': 10,  # Axis label font size
            'xtick.labelsize': 9,  # X tick label font size
            'ytick.labelsize': 9,  # Y tick label font size
            'legend.fontsize': 9,  # Legend font size
            'figure.titlesize': 14,  # Figure title font size
        }
    )
