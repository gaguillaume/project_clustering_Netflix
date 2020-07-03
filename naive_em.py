"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    densite = np.zeros([n, K])
    posterior = np.zeros([n, K])
    for i in range(K):
        # Densité gaussienne
        densite[:,i] = multivariate_normal.pdf(X, mixture.mu[i,:], mixture.var[i])
        # P(x_i|theta) - loi à posteriori
        posterior[:,i] = densite[:,i]*mixture.p[i]
    vraisemblance = np.sum(posterior, axis=1)
    post = posterior / vraisemblance[:,None]
    log_vraisemblance = np.log(posterior.sum(axis=1)).sum()
    return post, log_vraisemblance


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    K = post.shape[1]
    mu = np.zeros([K, d])
    sigma = np.zeros(K)
    # p est la moyenne des p(j|i)
    p = post.mean(axis=0)

    for i in range(K):
        mu[i,:] = (post[:,i] @ X)/post[:,i].sum()
        sigma[i] = ( (post[:,i] @ (X - mu[i,:])**2) / post[:,i].sum() ).mean()

    return GaussianMixture(mu, sigma, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    gauss_mixture = mixture
    old_log_vraisemblance = np.inf
    log_vraisemblance = 0
    while abs(old_log_vraisemblance - log_vraisemblance) > 1e-6*abs(log_vraisemblance):
        old_log_vraisemblance = log_vraisemblance
        post, log_vraisemblance = estep(X, gauss_mixture)
        gauss_mixture = mstep(X, post)
    return gauss_mixture, post, log_vraisemblance


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, _ = estep(X, mixture)
    n = X.shape[0]
    for i in range(n):
        X[i][X[i] == 0] = mixture.mu[np.argmax(post[i]), X[i]==0]
    return X