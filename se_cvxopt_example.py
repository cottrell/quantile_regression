# https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem/407478#407478
import io

import numpy as np
import pandas as pd
import requests
import os

filename = 'se_cvxopt_example.csv.gz'

if not os.path.exists(filename):
    url = "http://freakonometrics.free.fr/rent98_00.txt"
    s = requests.get(url).content
    base = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\t')
    base.to_csv(filename)

print(f'reading {filename} ...', end='')
base = pd.read_csv(filename)
print(' done')

tau = 0.3

from cvxopt import matrix, solvers

X = pd.DataFrame(columns=[0, 1])
X[1] = base["area"]  # data points for independent variable area
X[2] = base["yearc"]  # data points for independent variable year
X[0] = 1  # intercept

K = X.shape[1]
N = X.shape[0]

# equality constraints - left hand side

A1 = X.to_numpy()  # intercepts & data points - positive weights
A2 = X.to_numpy() * -1  # intercept & data points - negative weights
A3 = np.identity(N)  # error - positive
A4 = np.identity(N) * -1  # error - negative

A = np.concatenate((A1, A2, A3, A4), axis=1)  # all the equality constraints

# equality constraints - right hand side
b = base["rent_euro"].to_numpy()

# goal function - intercept & data points have 0 weights
# positive error has tau weight, negative error has 1-tau weight
c = np.concatenate((np.repeat(0, 2 * K), tau * np.repeat(1, N), (1 - tau) * np.repeat(1, N)))

# converting from numpy types to cvxopt matrix

Am = matrix(A)
bm = matrix(b)
cm = matrix(c)

# all variables must be greater than zero
# adding inequality constraints - left hand side
n = Am.size[1]
G = matrix(0.0, (n, n))
G[:: n + 1] = -1.0

# adding inequality constraints - right hand side (all zeros)
h = matrix(0.0, (n, 1))

# solving the model
sol = solvers.lp(cm, G, h, Am, bm, solver='glpk')

x = sol['x']

# both negative and positive components get values above zero, this gets fixed here
beta = x[0:K] - x[K : 2 * K]

print(beta)
