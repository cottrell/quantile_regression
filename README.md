# Quantile and CDF Regression Example

### Quantile Regression Objective
$$ J(\tau) = E\left(\rho(\tau, Y - u(\tau, X)|X\right)$$

### CDF Regression Objective
$$ J(y_c) = E\left(1_{Y < y_c} \log v(y_c, X) + (1 - 1_{Y < y_c}) \log(1 - v(y_x, X)) | X\right)$$

The functions $u$, $v$ must be monotonic in $\tau$ and $y_c$ respectively.

## Unconditional Distribution of $Y$

### Quantile Regression

<table>
<tr>
  <td><b>TensorFlow Implementation</b></td>
  <td><b>JAX Implementation</b></td>
</tr>
<tr>
  <td><img src="figs/tf/q_nox.png" width="600"></td>
  <td><img src="figs/jax/q_nox.png" width="600"></td>
</tr>
</table>

### CDF Estimation via Logistic Regression with Monotone Network

<table>
<tr>
  <td><b>TensorFlow Implementation</b></td>
  <td><b>JAX Implementation</b></td>
</tr>
<tr>
  <td><img src="figs/tf/p_nox.png" width="600"></td>
  <td><img src="figs/jax/p_nox.png" width="600"></td>
</tr>
</table>

## Conditional Distribution of $Y|X$

### Quantile Regression

<table>
<tr>
  <td><b>TensorFlow Implementation</b></td>
  <td><b>JAX Implementation</b></td>
</tr>
<tr>
  <td><img src="figs/tf/q.png" width="600"></td>
  <td><img src="figs/jax/q.png" width="600"></td>
</tr>
</table>

### CDF Estimation via Logistic Regression with Monotone Network

<table>
<tr>
  <td><b>TensorFlow Implementation</b></td>
  <td><b>JAX Implementation</b></td>
</tr>
<tr>
  <td><img src="figs/tf/p.png" width="600"></td>
  <td><img src="figs/jax/p.png" width="600"></td>
</tr>
</table>

## TODO

- do more quantitative error plots etc.
- consider CRPS (Continuous Ranked Probability Score) instead of CDF loss
- normalizing flows ... i.e. round trip cdf and quantile as consistency constraint ... where is this done, is it actually doing anything to include this constraint.
- randomize $\tau$ sampling during training in jax instead of grid ...
