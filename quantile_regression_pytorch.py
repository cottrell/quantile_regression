
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quantile_regression_common import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEPS,
    cdf_plot,
    cdf_plot_nox,
    get_data,
    loss_plot,
    q_plot,
    q_plot_nox,
    set_consistent_figure_params,
)

torch.set_default_dtype(torch.float64)

# Set up directories
_mydir = os.path.dirname(os.path.abspath(__file__))
_fig_dir = os.path.join(_mydir, "figs/pytorch")
os.makedirs(_fig_dir, exist_ok=True)


def logit(x):
    """Logit function that maps from (0, 1) to (-inf, inf)"""
    eps = 1e-6  # to avoid numerical issues
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


class NonNegDense(nn.Module):
    """Dense layer with non-negative weights for monotonicity constraints"""

    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation
        self.linear.weight.data.uniform_(0, 1)  # Initialize weights to be non-negative

    def forward(self, x):
        self.linear.weight.data.clamp_(0)  # Enforce non-negativity
        out = self.linear(x)
        return out if self.activation is None else self.activation(out)


class Dense(nn.Module):
    """Standard dense layer without non-negative constraint"""

    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(x)
        return out if self.activation is None else self.activation(out)


def rho_quantile_loss(tau, y, u):
    """Quantile loss function"""
    u = y.unsqueeze(1) - u.unsqueeze(0)
    tau = tau.unsqueeze(0)
    err = u * (tau - (u <= 0.0).float())
    return torch.mean(torch.sum(err, dim=1))


def logistic_loss(yc, y, u):
    """Logistic loss function for CDF estimation"""
    indicator = (y.unsqueeze(1) <= yc.unsqueeze(0)).float()
    log_u = torch.log(torch.clamp(u.unsqueeze(0), 1e-6, 1.0 - 1e-6))
    log_1mu = torch.log(torch.clamp(1.0 - u.unsqueeze(0), 1e-6, 1.0 - 1e-6))
    loss = indicator * log_u + (1 - indicator) * log_1mu
    return -torch.mean(torch.sum(loss, dim=1))


def crps_loss(yc, y, u):
    """CRPS loss for CDF estimation."""
    indicator = (y.unsqueeze(1) <= yc.unsqueeze(0)).float()
    u_bcast = u.unsqueeze(0)
    se = (u_bcast - indicator) ** 2
    weights = torch.ones_like(yc).view(1, -1, 1)
    loss = torch.mean(torch.sum(se * weights, dim=1))
    return loss


class QuantileNetworkNoX(nn.Module):
    """Deep quantile regression without covariates x"""

    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NonNegDense(
                    dims[i],
                    dims[i + 1],
                    activation=(torch.tanh if i < len(dims) - 2 else None),
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, tau):
        x = logit(tau)
        for layer in self.layers:
            x = layer(x)
        return x


class CDFNetworkNoX(nn.Module):
    """CDF estimation without covariates x"""

    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NonNegDense(
                    dims[i],
                    dims[i + 1],
                    activation=(
                        torch.tanh if i < len(dims) - 2 else torch.sigmoid
                    ),
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, yc):
        x = yc
        for layer in self.layers:
            x = layer(x)
        return x


class QuantileNetwork(nn.Module):
    """Deep quantile regression with covariates x"""

    def __init__(self, tau_dims, x_dims, final_dims):
        super().__init__()
        self.tau_layers = nn.ModuleList(
            [
                NonNegDense(
                    tau_dims[i],
                    tau_dims[i + 1],
                    activation=(
                        torch.tanh if i < len(tau_dims) - 2 else None
                    ),
                )
                for i in range(len(tau_dims) - 1)
            ]
        )
        self.x_layers = nn.ModuleList(
            [
                Dense(
                    x_dims[i],
                    x_dims[i + 1],
                    activation=(torch.tanh if i < len(x_dims) - 2 else None),
                )
                for i in range(len(x_dims) - 1)
            ]
        )
        self.final_layers = nn.ModuleList(
            [
                Dense(
                    final_dims[i],
                    final_dims[i + 1],
                    activation=None,
                )
                for i in range(len(final_dims) - 1)
            ]
        )

    def forward(self, tau, x):
        u = logit(tau)
        for layer in self.tau_layers:
            u = layer(u)

        v = x
        for layer in self.x_layers:
            v = layer(v)

        v_squared = v**2
        combined = v_squared.unsqueeze(1) * u.unsqueeze(0)

        q = combined
        for layer in self.final_layers:
            q = layer(q)

        return q


class CDFNetwork(nn.Module):
    """CDF estimation with covariates x"""

    def __init__(self, yc_dims, x_dims, final_dims, epsilon=1e-12):
        super().__init__()
        self.yc_layers = nn.ModuleList(
            [
                NonNegDense(
                    yc_dims[i],
                    yc_dims[i + 1],
                    activation=(
                        torch.tanh if i < len(yc_dims) - 2 else None
                    ),
                )
                for i in range(len(yc_dims) - 1)
            ]
        )
        self.x_layers = nn.ModuleList(
            [
                Dense(
                    x_dims[i],
                    x_dims[i + 1],
                    activation=(torch.tanh if i < len(x_dims) - 2 else None),
                )
                for i in range(len(x_dims) - 1)
            ]
        )
        self.final_layers = nn.ModuleList(
            [
                Dense(
                    final_dims[i],
                    final_dims[i + 1],
                    activation=(
                        torch.tanh if i < len(final_dims) - 2 else torch.sigmoid
                    ),
                )
                for i in range(len(final_dims) - 1)
            ]
        )
        self.epsilon = epsilon

    def forward(self, yc, x):
        u = yc
        for layer in self.yc_layers:
            u = layer(u)

        v = x
        for layer in self.x_layers:
            v = layer(v)

        v_squared = v**2
        combined = v_squared.unsqueeze(1) * u.unsqueeze(0)

        p = combined
        for layer in self.final_layers:
            p = layer(p)

        return self.epsilon + (1 - 2 * self.epsilon) * p


def sanity_plot_nox(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot quantile regression without covariates"""
    data = get_data()
    tau, y = torch.from_numpy(data["tau"]), torch.from_numpy(data["y"])

    model = QuantileNetworkNoX(dims=[1, 16, 16, 1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(steps):
        optimizer.zero_grad()
        u = model(tau)
        loss = rho_quantile_loss(tau, y, u)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    q = model(tau).detach().numpy().squeeze()

    loss_plot(losses, "qloss_nox", _fig_dir)
    q_plot_nox(y.numpy(), q, tau.numpy(), _fig_dir)


def cdfsanity_plot_nox(
    steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE, loss="logistic"
):
    """Train and plot CDF estimation without covariates"""
    loss_name = loss
    if loss == "logistic":
        loss_fn = logistic_loss
    elif loss == "crps":
        loss_fn = crps_loss
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    data = get_data()
    yc, y = torch.from_numpy(data["yc"]), torch.from_numpy(data["y"])

    model = CDFNetworkNoX(dims=[1, 16, 16, 1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(steps):
        optimizer.zero_grad()
        u = model(yc)
        loss = loss_fn(yc, y, u)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    cdf = model(yc).detach().numpy().squeeze()

    loss_plot(losses, f"cdfloss_nox_{loss_name}", _fig_dir)
    cdf_plot_nox(y.numpy(), cdf, yc.numpy(), _fig_dir, extra_name=loss_name)


def sanity_plot(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot quantile regression with covariates"""
    data = get_data()
    tau, y, x = (
        torch.from_numpy(data["tau"]),
        torch.from_numpy(data["y"]),
        torch.from_numpy(data["x"]),
    )
    sigma, mu = data["sigma"], data["mu"]

    model = QuantileNetwork(
        tau_dims=[1, 64, 64], x_dims=[1, 64, 64], final_dims=[64, 1]
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(steps):
        optimizer.zero_grad()
        u = model(tau, x)
        loss = rho_quantile_loss(tau, y, u)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    q = model(tau, x).detach().numpy().squeeze()

    loss_plot(losses, "qloss", _fig_dir)
    q_plot(x.numpy(), y.numpy(), mu, sigma, q, tau.numpy(), losses, _fig_dir)


def cdfsanity_plot(
    steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE, loss="logistic"
):
    """Train and plot CDF estimation with covariates"""
    loss_name = loss
    if loss == "logistic":
        loss_fn = logistic_loss
    elif loss == "crps":
        loss_fn = crps_loss
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    data = get_data()
    yc, y, x = (
        torch.from_numpy(data["yc"]),
        torch.from_numpy(data["y"]),
        torch.from_numpy(data["x"]),
    )
    sigma, mu = data["sigma"], data["mu"]

    model = CDFNetwork(
        yc_dims=[1, 64, 64], x_dims=[1, 64, 64], final_dims=[64, 1]
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(steps):
        optimizer.zero_grad()
        u = model(yc, x)
        loss = loss_fn(yc, y, u)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    cdf = model(yc, x).detach().numpy().squeeze()

    loss_plot(losses, f"cdfloss_{loss_name}", _fig_dir)
    cdf_plot(
        x.numpy(),
        y.numpy(),
        mu,
        sigma,
        cdf,
        yc.numpy(),
        losses,
        _fig_dir,
        extra_name=loss_name,
    )


if __name__ == "__main__":
    set_consistent_figure_params()
    plt.ion()
    sanity_plot_nox()
    cdfsanity_plot_nox(loss="logistic")
    cdfsanity_plot_nox(loss="crps")
    sanity_plot()
    cdfsanity_plot(loss="logistic")
    cdfsanity_plot(loss="crps")
