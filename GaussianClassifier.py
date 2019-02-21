import numpy as np
from statistics import cov
from Classifier import Classifier


class GaussianDistribution:
    def __init__(self, mu, comat=None, coeff=0.5, var=0.):
        self.mu = mu
        self.comat = comat
        self.inv = np.linalg.inv(comat) if not var else None
        self.coeff = coeff
        self.var = var
        self.denom = np.sqrt((2*np.pi)**len(mu) * self.comat) if not var else None

    # f ~ N(mu, comat)
    # f(x) = exp(-0.5(x-mu)T * inv(comat) * (x-mu)) / sqrt((2*pi)^k * det(comat))
    # But f(x) value is too small. for standardize, we ignore denominator(like evidence in Bayesian)
    def f(self, vec):
        vec = vec - self.mu
        if self.var:
            return np.sum(vec / self.var)
        return np.sum(np.exp(-self.coeff * vec.T.dot(self.inv).dot(vec)) / self.denom)


class GaussianClassifier(Classifier):
    name = 'Gaussian Classifier'

    def __init__(self, only_red=False, coeff=1e-3, prior=0):
        self.only_red = only_red
        self.dist_skin = None
        self.dist_non_skin = None
        self.coeff = coeff
        self.prior = prior

    def fit(self, X, y):
        X = np.copy(X)
        y = np.copy(y)
        X = X[:, 2].reshape(-1, 1) if self.only_red else X
        channel = range(X.shape[1])
        coeff = self.coeff

        skin = X[y == 1]
        non_skin = X[y == 2]
        if self.only_red and not self.prior:
            self.prior = skin.shape[0] / X.shape[0]

        if self.only_red:
            mu_skin = np.mean(skin)
            diff_skin = skin - mu_skin
            var_skin = np.sum(diff_skin ** 2)
            self.dist_skin = GaussianDistribution(mu_skin, var=var_skin)

            mu_non_skin = np.mean(non_skin)
            diff_non_skin = non_skin - mu_non_skin
            var_non_skin = np.sum(diff_non_skin ** 2)
            self.dist_non_skin = GaussianDistribution(mu_non_skin, var=var_non_skin)
        else:
            mu_skin = np.mean(skin, axis=0)
            diff_skin = X - mu_skin
            comat_skin = np.array([[cov(diff_skin[:, x], diff_skin[:, y])
                                    for x in channel] for y in channel])
            self.dist_skin = GaussianDistribution(mu_skin, comat_skin, coeff)

            mu_non_skin = np.mean(non_skin, axis=0)
            diff_non_skin = non_skin - mu_non_skin
            comat_non_skin = np.array([[cov(diff_non_skin[:, x], diff_non_skin[:, y])
                                        for x in channel] for y in channel])
            self.dist_non_skin = GaussianDistribution(mu_non_skin, comat_non_skin, coeff)

    def predict(self, X):
        return np.array([1 if self.__predict_elem(self.dist_skin, self.dist_non_skin, val)
                         else 2 for val in X])

    def evaluate(self, X, y):
        X = np.copy(X)
        y = np.copy(y)
        X = X[:, 2].reshape(-1, 1) if self.only_red else X

        pred_pos = [self.predict(X) == 1]
        real_pos = [y == 1]
        precision = np.sum(np.logical_and(pred_pos, real_pos)) / np.sum(pred_pos) if np.sum(pred_pos) else 0
        recall = np.sum(np.logical_and(pred_pos, real_pos)) / np.sum(real_pos) if np.sum(real_pos) else 0

        return precision, recall

    def __predict_elem(self, dist_skin, dist_non_skin, val):
        skin_prob = dist_skin.f(val)
        non_skin_prob = dist_non_skin.f(val)
        if self.only_red:
            skin_prob *= self.prior
            non_skin_prob *= 1 - self.prior
        return skin_prob > non_skin_prob  # Penalty for skin
