# https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem/407478#407478
#
# NOTE: This is quantile regression for linear regression problem. It turns into a linear programming problem.
# NOTE: 2023-01-16 I got this working. See do plot.
#

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

try:
    base
except NameError:
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

def solve_with_cvxopt():
    # solving the model
    sol = solvers.lp(cm, G, h, Am, bm, solver='glpk')

    x = sol['x']

    # both negative and positive components get values above zero, this gets fixed here
    beta = x[0:K] - x[K : 2 * K]

    print(beta)
    return locals()


from scipy.optimize import linprog
def solve_with_scipy():
    sol = linprog(cm, G, h, Am, bm)
    x = sol['x']
    # both negative and positive components get values above zero, this gets fixed here
    beta = x[0:K] - x[K : 2 * K]
    print(beta)
    return locals()


def plot_solution(beta, beta2=None):
    import plotly.graph_objects as go
    x = X.values[:, 1]
    y = X.values[:, 2]
    z = b
    zp = X.values @ beta
    traces = [
        go.Scatter3d(z=z, x=x, y=y, marker_size=3, mode='markers', opacity=0.50, marker_color='red', name='data'),
        go.Scatter3d(z=zp, x=x, y=y, opacity=0.50, mode='markers', marker_size=3, marker_color='blue', name='A'),
        go.Mesh3d(z=zp, x=x, y=y, opacity=0.50, color='green', name='Surface A'),
    ]
    if beta2 is not None:
        zp2 = X.values @ np.array(beta2).squeeze()
        traces.extend([
            go.Scatter3d(z=zp2, x=x, y=y, opacity=0.50, mode='markers', marker_size=3, marker_color='pink', name='B'),
            go.Mesh3d(z=zp2, x=x, y=y, opacity=0.50, color='purple', name='Surface B'),
        ])
        print(f'len traces {len(traces)}')

    fig = go.Figure(data=traces)
    fig.update_layout(title='quantile regression example', autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()
    return locals()


def do():
    a = solve_with_cvxopt()
    b = solve_with_cvxopt()
    plot_solution(a['beta'], b['beta'])
