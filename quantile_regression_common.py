import random
import os

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LEARNING_RATE = 0.005
DEFAULT_STEPS = 5_000

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


def cdf_plot(x, y, mu, sigma, cdf, yc, loss, _fig_dir):
    fig = plt.figure(1, figsize=(12, 6))
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

    X = np.repeat(x, cdf.shape[1], axis=1)
    Y = np.repeat(yc.T, cdf.shape[0], axis=0)
    ax[1].contour(X, Y, cdf)
    ax[1].set_xlabel(f"x[:,0] (x.shape={x.shape})")
    ax[1].set_title('inferred cdf (contour plot)')
    fig.tight_layout()

    fig2 = plt.figure(2, figsize=(12, 6))
    ax = fig2.gca()
    ax.semilogy(loss)

    fig_path = os.path.join(_fig_dir, 'p.png')
    plt.figure(1)
    plt.savefig(fig_path)
    plt.show()
    return locals()

def q_plot(x, y, mu, sigma, q, tau, loss, _fig_dir):
    fig = plt.figure(1, figsize=(12, 6))
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
    ax[1].plot(x[:, 0], q, alpha=0.5)
    ax[1].set_xlabel(f"x[:,0] (x.shape={x.shape})")
    ax[1].set_title('inferred quantiles')
    fig.tight_layout()

    fig2 = plt.figure(2, figsize=(12, 6))
    ax = fig2.gca()
    ax.semilogy(loss)

    fig_path = os.path.join(_fig_dir, 'q.png')
    plt.figure(1)
    plt.savefig(fig_path)
    plt.show()
    return locals()


def cdf_plot_nox(y, cdf, yc, _fig_dir):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.subplots(1, 1)
    n = len(y)
    p = np.linspace(1.0 / n, 1 - 1.0 / n, n)
    i = y[:, 0].argsort()
    ax.plot(p, y[i, 0], ".", label="data", alpha=0.5)

    ax.plot(cdf, yc, "g.-", label='fit', linewidth=2)
    ax.legend()
    ax.set_xlabel("$P(Y < y)$")
    ax.set_ylabel('y')
    ax.set_title('CDF')
    fig.tight_layout()
    fig_path = os.path.join(_fig_dir, 'p_nox.png')
    plt.savefig(fig_path)
    plt.show()


def q_plot_nox(y, q, tau, _fig_dir):

    fig = plt.figure(1)
    fig.clf()
    ax = fig.subplots(1, 1)
    n = len(y)
    p = np.linspace(1.0 / n, 1 - 1.0 / n, n)
    i = y[:, 0].argsort()
    ax.plot(p, y[i, 0], ".", label="data", alpha=0.5)

    ax.plot(tau, q, "g.-", label='fit', linewidth=2)
    ax.legend()
    ax.set_xlabel("tau: $P(Y < y)$")
    ax.set_ylabel('y')
    ax.set_title('quantile')
    fig.tight_layout()
    fig_path = os.path.join(_fig_dir, 'q_nox.png')
    plt.savefig(fig_path)
    plt.show()


def loss_plot(loss, name, _fig_dir):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    ax.semilogy(loss)
    fig_path = os.path.join(_fig_dir, f'{name}.png')
    plt.savefig(fig_path)
    plt.show()