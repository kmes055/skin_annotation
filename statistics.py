import numpy as np


def cov(x, y):
    Sxy, Sxx, Syy = 0, 0, 0

    for b, s in zip(x, y):
        Sxy += b * s
        Sxx += b ** 2
        Syy += s ** 2

    return Sxy / np.sqrt(Sxx * Syy)


def pdf(x, distribute):
    pass


def mahalanobis_distance(x, mu, mat):
    mat = np.linalg.inv(mat)
    x = x - mu
    return np.sqrt(x.T.dot(mat).dot(x))
