import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from quantile_regression_common import get_data, set_consistent_figure_params, cdf_plot, q_plot, cdf_plot_nox, q_plot_nox, loss_plot, DEFAULT_LEARNING_RATE, DEFAULT_STEPS

# NOTE: WARNING: 2025-05-06 produced converting from tf to jax using claude etc lightly reviewed

# Set up directories
_mydir = os.path.dirname(os.path.abspath(__file__))
_fig_dir = os.path.join(_mydir, 'figs/jax')
os.makedirs(_fig_dir, exist_ok=True)



def logit(x):
    """Logit function that maps from (0, 1) to (-inf, inf)"""
    eps = 1e-6  # to avoid numerical issues
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)


class NonNegDense(eqx.Module):
    """Dense layer with non-negative weights for monotonicity constraints"""

    weight: jax.Array
    bias: jax.Array
    activation: callable = eqx.static_field()

    def __init__(self, in_size, out_size, *, key, activation):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.uniform(wkey, (out_size, in_size))  # unconstrained
        self.bias = jax.random.uniform(bkey, (out_size,))
        self.activation = activation

    def __call__(self, x):
        w = jnp.abs(self.weight)  # constraint enforced here
        out = x @ w.T + self.bias
        return out if self.activation is None else self.activation(out)


class Dense(eqx.Module):
    """Standard dense layer without non-negative constraint"""

    weight: jax.Array
    bias: jax.Array
    activation: callable = eqx.static_field()

    def __init__(self, in_size, out_size, *, key, activation):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size)) * 0.1
        self.bias = jnp.zeros((out_size,))
        self.activation = activation

    def __call__(self, x):
        out = x @ self.weight.T + self.bias
        return out if self.activation is None else self.activation(out)


def rho_quantile_loss(tau, y, u):
    """Quantile loss function"""
    # y: (n, 1), u: (n_tau, 1)
    # output: scalar
    u = y[:, None] - u[None, :, :]  # shape (n, n_tau, 1)
    tau = tau[None, :, :]  # shape (1, n_tau, 1)
    err = u * (tau - (u <= 0.0).astype(jnp.float32))
    return jnp.mean(jnp.sum(err, axis=(1, 2)))  # sum over tau, then mean over batch


def logistic_loss(yc, y, u):
    """Logistic loss function for CDF estimation"""
    # y: (n, 1), yc: (n_yc, 1), u: (n_yc, 1)
    # u is P(Y <= yc)
    # output: scalar
    indicator = (y[:, None, :] <= yc[None, :, :]).astype(jnp.float32)
    log_u = jnp.log(jnp.clip(u[None, :, :], 1e-6, 1.0 - 1e-6))
    log_1mu = jnp.log(jnp.clip(1.0 - u[None, :, :], 1e-6, 1.0 - 1e-6))
    loss = indicator * log_u + (1 - indicator) * log_1mu
    return -jnp.mean(jnp.sum(loss, axis=(1, 2)))


class QuantileNetworkNoX(eqx.Module):
    """Deep quantile regression without covariates x"""

    layers: list

    def __init__(self, dims, *, key):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            NonNegDense(dims[i], dims[i + 1], key=keys[i], activation=(jax.nn.tanh if i < len(dims) - 2 else None))
            for i in range(len(dims) - 1)
        ]

    def __call__(self, tau):
        x = logit(tau)
        for layer in self.layers:
            x = layer(x)
        return x


class CDFNetworkNoX(eqx.Module):
    """CDF estimation without covariates x"""

    layers: list

    def __init__(self, dims, *, key):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            NonNegDense(
                dims[i], dims[i + 1], key=keys[i], activation=(jax.nn.tanh if i < len(dims) - 2 else jax.nn.sigmoid)
            )
            for i in range(len(dims) - 1)
        ]

    def __call__(self, yc):
        x = yc
        for layer in self.layers:
            x = layer(x)
        return x


class QuantileNetwork(eqx.Module):
    """Deep quantile regression with covariates x"""

    tau_layers: list
    x_layers: list
    final_layers: list

    def __init__(self, tau_dims, x_dims, final_dims, *, key):
        keys = jax.random.split(key, 3)

        # Tau network layers
        tau_keys = jax.random.split(keys[0], len(tau_dims) - 1)
        self.tau_layers = [
            Dense(
                tau_dims[i],
                tau_dims[i + 1],
                key=tau_keys[i],
                activation=(jax.nn.tanh if i < len(tau_dims) - 2 else None),
            )
            for i in range(len(tau_dims) - 1)
        ]

        # X network layers
        x_keys = jax.random.split(keys[1], len(x_dims) - 1)
        self.x_layers = [
            Dense(x_dims[i], x_dims[i + 1], key=x_keys[i], activation=(jax.nn.tanh if i < len(x_dims) - 2 else None))
            for i in range(len(x_dims) - 1)
        ]

        # Final layers
        final_keys = jax.random.split(keys[2], len(final_dims) - 1)
        self.final_layers = [
            Dense(
                final_dims[i],
                final_dims[i + 1],
                key=final_keys[i],
                activation=(jax.nn.tanh if i < len(final_dims) - 2 else None),
            )
            for i in range(len(final_dims) - 1)
        ]

    def __call__(self, tau, x):
        # Process tau through tau network
        u = logit(tau)
        for layer in self.tau_layers:
            u = layer(u)

        # Process x through x network
        v = x
        for layer in self.x_layers:
            v = layer(v)

        # Square v to ensure positivity
        v_squared = v**2

        # Combine features
        # Handle broadcasting for batched inputs
        batched_u = u
        batched_v = v_squared

        if len(v_squared.shape) > 1 and len(u.shape) > 1:
            # Both inputs have batches
            if u.shape[0] == 1:
                # u has single batch, broadcast against v
                combined = batched_v * batched_u
            elif v_squared.shape[0] == 1:
                # v has single batch, broadcast against u
                combined = batched_v * batched_u
            else:
                # Both have multiple batches, need explicit handling
                # Create a grid of combinations (inefficient but explicit)
                combined = v_squared[:, None, :] * u[None, :, :]
        else:
            # Simple case - no complex batching
            combined = v_squared * u

        # Process through final layers
        q = combined
        for layer in self.final_layers:
            q = layer(q)

        return q


class CDFNetwork(eqx.Module):
    """CDF estimation with covariates x"""

    yc_layers: list
    x_layers: list
    final_layers: list
    epsilon: float

    def __init__(self, yc_dims, x_dims, final_dims, *, key, epsilon=1e-12):
        keys = jax.random.split(key, 3)

        # YC network layers - use nonnegative for monotonicity
        yc_keys = jax.random.split(keys[0], len(yc_dims) - 1)
        self.yc_layers = [
            NonNegDense(
                yc_dims[i], yc_dims[i + 1], key=yc_keys[i], activation=(jax.nn.tanh if i < len(yc_dims) - 2 else None)
            )
            for i in range(len(yc_dims) - 1)
        ]

        # X network layers
        x_keys = jax.random.split(keys[1], len(x_dims) - 1)
        self.x_layers = [
            Dense(x_dims[i], x_dims[i + 1], key=x_keys[i], activation=(jax.nn.tanh if i < len(x_dims) - 2 else None))
            for i in range(len(x_dims) - 1)
        ]

        # Final layers with sigmoid activation at the end
        final_keys = jax.random.split(keys[2], len(final_dims) - 1)
        self.final_layers = [
            Dense(
                final_dims[i],
                final_dims[i + 1],
                key=final_keys[i],
                activation=(jax.nn.tanh if i < len(final_dims) - 2 else jax.nn.sigmoid),
            )
            for i in range(len(final_dims) - 1)
        ]

        self.epsilon = epsilon

    def __call__(self, yc, x):
        # Process yc through yc network
        u = yc
        for layer in self.yc_layers:
            u = layer(u)

        # Process x through x network
        v = x
        for layer in self.x_layers:
            v = layer(v)

        # Square v to ensure positivity
        v_squared = v**2

        # Combine features with same broadcasting logic as QuantileNetwork
        batched_u = u
        batched_v = v_squared

        if len(v_squared.shape) > 1 and len(u.shape) > 1:
            if u.shape[0] == 1:
                combined = batched_v * batched_u
            elif v_squared.shape[0] == 1:
                combined = batched_v * batched_u
            else:
                combined = v_squared[:, None, :] * u[None, :, :]
        else:
            combined = v_squared * u

        # Process through final layers
        p = combined
        for layer in self.final_layers:
            p = layer(p)

        # Ensure output is in (epsilon, 1-epsilon)
        return self.epsilon + (1 - 2 * self.epsilon) * p


@eqx.filter_jit
def make_quantile_step(model, opt_state, tau, y, opt):
    """JIT-compiled training step for quantile model without x"""

    def loss_fn(model):
        u = model(tau)
        return rho_quantile_loss(tau, y, u)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def make_cdf_step(model, opt_state, yc, y, opt):
    """JIT-compiled training step for CDF model without x"""

    def loss_fn(model):
        u = model(yc)
        return logistic_loss(yc, y, u)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def make_quantile_x_step(model, opt_state, tau, y, x, opt):
    """JIT-compiled training step for quantile model with x"""

    def loss_fn(model):
        u = model(tau, x)
        return rho_quantile_loss(tau, y, u)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def make_cdf_x_step(model, opt_state, yc, y, x, opt):
    """JIT-compiled training step for CDF model with x"""

    def loss_fn(model):
        u = model(yc, x)
        return logistic_loss(yc, y, u)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def sanity_plot_nox(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot quantile regression without covariates"""
    data = get_data()
    tau, y = data["tau"], data["y"]

    key = jax.random.PRNGKey(0)
    model = QuantileNetworkNoX(dims=[1, 16, 16, 1], key=key)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    losses = []
    for i in range(steps):
        model, opt_state, loss = make_quantile_step(model, opt_state, tau, y, optimizer)
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    q = jax.vmap(model)(tau).squeeze()

    loss_plot(losses, 'qloss_nox', _fig_dir)

    q_plot_nox(y, q, tau, _fig_dir)

def cdfsanity_plot_nox(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot CDF estimation without covariates"""
    data = get_data()
    yc, y = data["yc"], data["y"]

    key = jax.random.PRNGKey(0)
    model = CDFNetworkNoX(dims=[1, 16, 16, 1], key=key)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    losses = []
    for i in range(steps):
        model, opt_state, loss = make_cdf_step(model, opt_state, yc, y, optimizer)
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    cdf = jax.vmap(model)(yc).squeeze()

    loss_plot(losses, 'cdfloss_nox', _fig_dir)

    cdf_plot_nox(y, cdf, yc, _fig_dir)



def sanity_plot(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot quantile regression with covariates"""
    data = get_data()
    tau, y, x = data["tau"], data["y"], data["x"]
    sigma, mu = data["sigma"], data["mu"]

    key = jax.random.PRNGKey(0)
    model = QuantileNetwork(tau_dims=[1, 64, 64], x_dims=[1, 64, 64], final_dims=[64, 1], key=key)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    losses = []
    for i in range(steps):
        model, opt_state, loss = make_quantile_x_step(model, opt_state, tau, y, x, optimizer)
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")

    q = model(tau, x).squeeze()

    loss_plot(losses, 'qloss', _fig_dir)

    q_plot(x, y, mu, sigma, q, tau, losses, _fig_dir)


def cdfsanity_plot(steps=DEFAULT_STEPS, learning_rate=DEFAULT_LEARNING_RATE):
    """Train and plot CDF estimation with covariates"""
    data = get_data()
    yc, y, x = data["yc"], data["y"], data["x"]
    sigma, mu = data["sigma"], data["mu"]
    key = jax.random.PRNGKey(0)
    model = CDFNetwork(yc_dims=[1, 64, 64], x_dims=[1, 64, 64], final_dims=[64, 1], key=key)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    for i in range(steps):
        model, opt_state, loss = make_cdf_x_step(model, opt_state, yc, y, x, optimizer)
        if i % 1 == 0:
            losses.append(loss.item())
            print(f"Step {i}, Loss: {loss.item():.4f}")
    cdf = model(yc, x).squeeze()
    loss_plot(losses, 'cdfloss', _fig_dir)
    cdf_plot(x, y, mu, sigma, cdf, yc, losses, _fig_dir)


if __name__ == '__main__':
    set_consistent_figure_params()
    plt.ion()
    sanity_plot_nox()
    cdfsanity_plot_nox()
    sanity_plot()
    cdfsanity_plot()
