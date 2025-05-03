# Quantile and CDF Regression Example

Quantile regression objective

$$ J(\tau) = E\left(\rho(\tau, Y - u(\tau, X)|X\right)$$

<img src="https://render.githubusercontent.com/render/math?math=J(\tau) = E\left(\rho(\tau, Y - u(\tau, X)|X\right)">

CDF regression objective


$$ J(y_c) = E\left(1_{Y < y_c} \log v(y_c, X) + (1 - 1_{Y < y_c}) \log(1 - v(y_x, X)) | X\right)$$

<img src="https://render.githubusercontent.com/render/math?math=J(y_c) = E\left(\mathbb{1}_{Y < y_c} \log v(y_c, X) + (1 - \mathbb{1}_{Y < y_c}) \log(1 - v(y_x, X)) | X\right)">

The functions $u$, $v$ must be monotonic in $\tau$ and $y_c$ respectively.

## Unconditional distribution of $Y$

### Quantile regression

<img src="q_nox.png" alt="unconditional quantile regression" width="800">

### CDF estimation via logistic regression with monotone network

<img src="p_nox.png" alt="unconditional cdf regression" width="800">

## Conditional distributional of $Y|X$

### Quantile regression

<img src="q.png" alt="conditional quantile regression" width="800">

### CDF estimation via logistic regression with monotone network

<img src="p.png" alt="conditional cdf regression" width="800">

# TODO

Do more quantitative error plots etc.
