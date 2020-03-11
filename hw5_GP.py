import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
from scipy.optimize import minimize
import math
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--optimize", type=int, default=1)
args = parser.parse_args()


def read_dataset(filename="input.data"):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = np.array([line.strip().split()
                      for line in lines], dtype="float64")
    return lines


def RationalQuadraticKernel(xa, xb, alpha=10, l=1.0, var=5):  # alpha=10, l=1

    return var * (1+cdist(xa, xb, 'sqeuclidean')/2/alpha/l/l)**(-alpha)


def visualize(mu, upper, lower):
    plt.xlim(-60, 60)
    plt.scatter(X, Y, c='r', edgecolors='face')
    plt.plot(x_test.ravel(), mu.ravel(), 'b')
    plt.fill_between(x_test.ravel(), upper, lower, alpha=0.3)
    plt.show()


def marginal_likelihood(theta):
    C = RationalQuadraticKernel(X, X, alpha=theta[0], l=theta[1])
    return 0.5*Y.T.dot(np.linalg.inv(C)).dot(Y) + 0.5*math.log(np.linalg.det(C)) + n/2*math.log(2.0*math.pi)


def GP(alpha=10, l=1.0):
    C = RationalQuadraticKernel(X, X, alpha, l)
    kstar = np.add(RationalQuadraticKernel(
        x_test, x_test, alpha, l), np.eye(len(x_test))*1.0/beta)
    kxx = RationalQuadraticKernel(X, x_test, alpha, l)

    mu = kxx.T.dot(np.linalg.inv(C)).dot(Y)  # 500,1
    var = kstar - kxx.T.dot(np.linalg.inv(C)).dot(kxx)  # 500,500
    upper = np.zeros(points)
    lower = np.zeros(points)
    for i in range(points):
        upper[i] = mu[i, 0] + var[i, i]*1.96
        lower[i] = mu[i, 0] - var[i, i]*1.96

    visualize(mu, upper, lower)


if __name__ == '__main__':
    data = read_dataset()
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)
    points = 10000
    x_test = np.linspace(-60, 60, points).reshape(-1, 1)
    beta = 5
    n, _ = X.shape

    if args.optimize is 0:
        GP()

    elif args.optimize is 1:
        theta = [1.0, 1.0]
        res = minimize(marginal_likelihood, theta)

        alhpa_min, l_min = res.x
        GP(alhpa_min, l_min)
