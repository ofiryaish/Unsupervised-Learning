import random
import numpy as np

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def multivariate_normal_posterior(liklihood_xk_wis, prior_wis):
    """
    P(ωi|xk,ˆμ)
    :param liklihood_xk_wis: p(xk|ωi, ˆμ) for all i's - C-size vector where C is the number of clusters. It's just the PDF
    :param prior_wis: p(wi) for all i's - C-size vector where C is the number of clusters
    :return: the posterior for all i's
    """
    return (liklihood_xk_wis*prior_wis)/np.sum(liklihood_xk_wis*prior_wis)


def multivariate_normal_posterior_is_xks(mus, sigmas, xks, prior_wis):
    """
    for each cluster (i), compute the posterior given each datapoint (xk):
    :param mus: Cluster means CxK
    :param sigmas: Cluster covariances CxKxK
    :param xks: datapints NXK
    param prior_wis: p(wi) for all i's - C-size vector
    :return: the posterior given each datapoint (xk) for each cluster (i) - CxN
    """
    likelihood_wis_xks = np.array([multivariate_normal.pdf(xks, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    posterior_wis_xks = np.apply_along_axis(multivariate_normal_posterior, 0, likelihood_wis_xks, prior_wis)
    return posterior_wis_xks


def multivariate_normal_mu_i_new(old_posterior_i_xks, xks):
    """
    compute the new mean acording to MLE algorithm:
    :param old_posterior_i_xks: The poserior of wi for all datapoints sks N-size vector
    :param xks: datapints NxK
    :return: the new mean for cluster i - K-size vector
    """
    if xks.ndim == 1:
        xks = np.expand_dims(xks, axis=xks.ndim)
    return np.sum(old_posterior_i_xks[:, np.newaxis]*xks, axis=0)/np.sum(old_posterior_i_xks)


def multivariate_normal_prior_i_new(old_posterior_i_xks):
    """
    compute the new prior acording to MLE algorithm:
    :param old_posterior_i_xks: The poserior of wi for all datapoints sks N-size vector
    :return: the new prior for cluster i - scalar
    """
    return np.mean(old_posterior_i_xks)


def multivariate_normal_sigma_i_new(old_posterior_i_xks, mu_i, xks):
    """
    compute the new covariance acording to MLE algorithm:
    :param old_posterior_i_xks: The poserior of wi for all datapoints xks N-size vector
    :param mu_i: the mean of cluster i K-size vector
    :param xks: datapints NxK
    :return: the new covariance for cluster i - KxK
    """
#     xks = np.expand_dims(xks, axis=xks.ndim)
#     mu_i = np.expand_dims(mu_i, axis=mu_i.ndim)
#     old_posterior_i_xks = np.expand_dims(old_posterior_i_xks, axis=[old_posterior_i_xks.ndim, old_posterior_i_xks.ndim + 1])
#     weighted_covariance_components = old_posterior_i_xks*np.vectorize(lambda xk, mu_i: (xk-mu_i)@((xk-mu_i)).T, signature='(k,1),(k,1)->(k,k)')(xks, mu_i)
#     return np.sum(weighted_covariance_components, axis=0)/np.sum(old_posterior_i_xks)
    weighted_covariance_components = old_posterior_i_xks[:, np.newaxis, np.newaxis] * (xks - mu_i)[:, :, np.newaxis] @ (xks - mu_i)[:, np.newaxis, :] 
    return np.sum(weighted_covariance_components, axis=0)/np.sum(old_posterior_i_xks)


def mle_step_mu(xks, posterior_wis_xks):
    """
    MLE step to compute the means
    :param xks: NxK
    :param posterior_wis_xks: posterior given each datapoint (xk) for each cluster (i) - CxN
    """
    new_mus = np.apply_along_axis(multivariate_normal_mu_i_new, 1, posterior_wis_xks, xks)
    return new_mus


def mle_step_prior(posterior_wis_xks):
    """
    MLE step to compute the prior probabilities
    :param posterior_wis_xks: posterior given each datapoint (xk) for each cluster (i) - CxN
    """
    new_prior = np.apply_along_axis(multivariate_normal_prior_i_new, 1, posterior_wis_xks)
    return new_prior


def mle_step_sigma(mus, xks, posterior_wis_xks):
    """
    MLE step to compute the covariance
    :param mus:  means - CxK
    :param xks: NxK
    :param posterior_wis_xks: posterior given each datapoint (xk) for each cluster (i) - CxN
    """
    if mus.ndim == 1:
        mus = np.expand_dims(mus, axis=mus.ndim)
    if xks.ndim == 1:
        xks = np.expand_dims(xks, axis=xks.ndim)
    new_sigmas = np.vectorize(multivariate_normal_sigma_i_new, signature="(N),(K),(N,K)->(K,K)")(posterior_wis_xks, mus, xks)
    return np.squeeze(new_sigmas)


def mle_init(xks, mus, sigmas, prior_wis, c=None, kmeans=False):
    if c is None:
        raise ValueError("unkown number of clusters is not suported yet")
    mle_type = [False, False, False]
    if mus is None:
        if kmeans:
            kmeans = KMeans(n_clusters=c) #random_state=0
            kmeans.fit(xks)
            mus = kmeans.cluster_centers_
        else:
            indices = random.sample(range(len(xks)), c)
            mus = xks[indices]
        mle_type[0] = True
    if sigmas is None:
        sigmas = np.array([np.eye(xks.shape[-1] if xks.ndim > 1 else 1)] * c)
        mle_type[1] = True
    if prior_wis is None:
        prior_wis = np.ones(c)/c
        mle_type[2] = True
    return mus, sigmas, prior_wis, mle_type


def mle(xks, mus=None, sigmas=None, prior_wis=None, c=None, max_iter=500, eps=1e-8, kmeans=False):
    log_likelihood_losses = []
    mus, sigmas, prior_wis, mle_type = mle_init(xks, mus, sigmas, prior_wis, c, kmeans)
    for i in range(max_iter):
        posterior_wis_xks = multivariate_normal_posterior_is_xks(mus, sigmas, xks, prior_wis)
        
        mus_new = mle_step_mu(xks, posterior_wis_xks) if mle_type[0] else mus
        sigmas_new = mle_step_sigma(mus, xks, posterior_wis_xks) if mle_type[1] else sigmas
        prior_wis_new = mle_step_prior(posterior_wis_xks) if mle_type[2] else prior_wis
         
        # likelihood = sum_{k=1^n}ln(sum_{j=1^c}p(x_k|w_j, theta_j)*p(w_j))
        likelihood_wis_xks = np.array([multivariate_normal.pdf(xks, mu, sigma) for mu, sigma in zip(mus_new, sigmas_new)])
        total_log_likelihood = np.sum(np.log((likelihood_wis_xks*prior_wis_new[:,np.newaxis]).sum(axis=0)))
        
        
        if log_likelihood_losses and np.abs(log_likelihood_losses[-1] - total_log_likelihood) < eps*np.abs(log_likelihood_losses[-1]):
            print("stop iteration after", i, "iterations due to convergence")
            break
        else:
            mus, sigmas, prior_wis = mus_new, sigmas_new, prior_wis_new
            log_likelihood_losses.append(total_log_likelihood)
    print("likelihood:", total_log_likelihood)
    return log_likelihood_losses, posterior_wis_xks, mus, sigmas, prior_wis