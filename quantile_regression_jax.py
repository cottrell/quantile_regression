import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt


def logit(x):
    eps = 1e-6  # to avoid numerical issues
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)


class NonNegDense(eqx.Module):
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



class QuantileNetworkNoX(eqx.Module):
    layers: list

    def __init__(self, dims, *, key):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            NonNegDense(dims[i], dims[i + 1], key=keys[i], activation=(jax.nn.tanh if i < len(dims)-2 else None))
            for i in range(len(dims) - 1)
        ]

    def __call__(self, tau):
        x = logit(tau)
        for layer in self.layers:
            x = layer(x)
        return x


def rho_quantile_loss(tau, y, u):
    # y: (n, 1), u: (n_tau, 1)
    # output: scalar
    u = y[:, None] - u[None, :, :]  # shape (n, n_tau, 1)
    tau = tau[None, :, :]  # shape (1, n_tau, 1)
    err = u * (tau - (u <= 0.0).astype(jnp.float64))
    return jnp.mean(jnp.sum(err, axis=(1, 2)))  # sum over tau, then mean over batch


def get_data(seed=0, m=250, n_tau=11):
    key = jax.random.PRNGKey(seed)
    x = (2 * jax.random.uniform(key, (m, 1)) - 1) * 2
    sigma = 0.4 * (1 + 5 / (10 * x[:, 0] ** 2 + 1))[:, None]
    mu = x ** 2 + 0.3 * x
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (m, 1))
    y = mu + sigma * z
    yc = jnp.linspace(jnp.min(y), jnp.max(y), n_tau)[:, None]
    tau = jnp.linspace(1.0 / n_tau, 1 - 1.0 / n_tau, n_tau)[:, None]
    return tau, y


@eqx.filter_jit
def make_step(model, opt_state, tau, y, opt):
    def loss_fn(model):
        u = model(tau)
        return rho_quantile_loss(tau, y, u)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_and_plot(steps=1000):
    tau, y = get_data()
    key = jax.random.PRNGKey(0)
    model = QuantileNetworkNoX(dims=[1, 16, 16, 1], key=key)
    opt = optax.adam(0.01)
    opt_state = opt.init(model)

    losses = []
    for _ in range(steps):
        model, opt_state, loss = make_step(model, opt_state, tau, y, opt)
        losses.append(loss)

    plt.figure()
    n = len(y)
    p = jnp.linspace(1.0 / n, 1 - 1.0 / n, n)
    i = jnp.argsort(y[:, 0])
    plt.plot(p, y[i, 0], ".", label="data", alpha=0.5)
    q = model(tau).squeeze()
    plt.plot(tau.squeeze(), q, "g.-", label="fit")
    plt.legend()
    plt.xlabel("tau")
    plt.ylabel("y")
    plt.title("quantile")
    plt.show()
    return model


if __name__ == "__main__":
    train_and_plot()
