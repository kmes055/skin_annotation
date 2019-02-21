import numpy as np
from Classifier import Classifier


class BayesianClassifier(Classifier):
    name = 'Bayesian classifier'

    def __init__(self, scale=10, only_red=False, prior=0):
        self.only_red = only_red
        self.likelihood = np.array([])
        self.prior = prior
        self.scale = scale

    def fit(self, X, y):
        X = np.copy(X)
        y = np.copy(y)
        X = X[:, 2].reshape(-1, 1) if self.only_red else X

        scale = self.scale
        X //= scale
        unit = 255 // scale + 1

        skin = X[y == 1].reshape((-1, X.shape[1]))
        non_skin = X[y == 2].reshape((-1, X.shape[1]))

        if not self.prior:
            self.prior = skin.shape[0] / X.shape[0]

        self.likelihood = np.zeros((2 * X.shape[1], unit))
        for i in range(X.shape[1]):
            self.likelihood[2*i, :np.max(skin[:, i]) + 1] = np.bincount(skin[:, i]) / skin.shape[0]
            self.likelihood[2*i + 1, :np.max(non_skin[:, i]) + 1] = np.bincount(non_skin[:, i]) / non_skin.shape[0]

        return self

    def predict(self, X):
        X //= self.scale
        return np.array([1 if self.__predict_elem(self.likelihood.T, val, self.prior)
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

    def __predict_elem(self, likelihood, val, prior):
        # posterior = likelihood * prior / evidence

        skin_prob = 1
        non_skin_prob = 1

        for i in range(val.shape[0]):
            skin_prob *= likelihood[val[i], 2*i]*prior
            non_skin_prob *= likelihood[val[i], 2*i + 1]*(1 - prior)

        return skin_prob > non_skin_prob

