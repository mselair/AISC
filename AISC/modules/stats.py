import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_gen, multivariate_normal_frozen

class multivariate_normal_(multivariate_normal_frozen):
    def __init__(self, X, allow_singular=False, seed=None):
        X = X.copy().T
        cov = np.cov(X.T)
        mu = X.mean(axis=0)
        super().__init__(mu, cov, allow_singular, seed)


    def pdf(self, X):
        X = X.copy().T
        return super().pdf(X)

