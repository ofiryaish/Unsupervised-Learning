import numpy as np
from HW.kmeans import kmeans


def find_centroids(memberships, xks, fuzziness):
    """
     - memberships is a matrix of dimensions nXk.
     - x is a matrix of samples  of dimensions nXm. Each sample is a row, and the number of columns is the number of features.
     - fuzziness is an integer parameter of the algorithm.
     """
    n, m = xks.shape
    k = memberships.shape[1]
    centroids = np.zeros((k, m))
    z = np.sum(memberships ** fuzziness, axis=0)
    for i in range(k):
        for j in range(n):
            centroids[i] += (memberships[j, i]**fuzziness) * xks[j] / z[i]
    return centroids


def fuzzy_covariance(memberships_i_xks, centroid_i, xks):
    """
    compute the fuzzy covariance acording to MLE algorithm:
    :param memberships_i_xks: The memberships of cntroid i for all datapoints xks N-size vector
    :param centroid_i: the centroid of cluster i K-size vector
    :param xks: datapints NxK
    :return: the new covariance for cluster i - KxK
    """
#     xks = np.expand_dims(xks, axis=xks.ndim)
#     centroid_i = np.expand_dims(centroid_i, axis=centroid_i.ndim)
#     memberships_i_xks = np.expand_dims(memberships_i_xks, axis=[memberships_i_xks.ndim, memberships_i_xks.ndim + 1])
#     weighted_covariance_components = memberships_i_xks*np.vectorize(
#         lambda xk, centroid_i: (centroid_i-xk) @ ((centroid_i-xk).T), signature='(k,1),(k,1)->(k,k)')(xks, centroid_i)
    weighted_covariance_components = memberships_i_xks[:, np.newaxis, np.newaxis] * (centroid_i - xks)[:, :, np.newaxis] @ (centroid_i - xks)[:, np.newaxis, :] 
    return np.sum(weighted_covariance_components, axis=0)/np.sum(memberships_i_xks)


def find_memberships(xks, centroids, fuzziness, memberships, distance_type, add_imaginary_center=False):
    """
     - x is a matrix of samples. Each sample is a row, and the number of columns is the number of features.
     - centroids is a matrix of centroids. Each centroid is a row, and the number of columns is the number of features.
     - fuzziness is an integer parameter of the algorithm.
     """
    n = xks.shape[0]
    k = centroids.shape[0]
    d = np.zeros((n, k+1 if add_imaginary_center else k))
    for i in range(k):
        if distance_type == "euclidean":
            d[:, i] = np.sum((xks - centroids[i]) ** 2, axis=1)
        if distance_type == "exponential":
            cluster_i_sum_of_memberships = memberships[:, i].mean()
            fuzzy_covariance_matrix = fuzzy_covariance(memberships[:, i], centroids[i], xks)
            fuzzy_hypervolume = np.linalg.det(fuzzy_covariance_matrix) ** 0.5
            d[:, i] = np.squeeze(np.apply_along_axis(lambda xk: (fuzzy_hypervolume/cluster_i_sum_of_memberships) * np.exp(
                (0.5 * (centroids[i]-xk).T @ np.linalg.inv(fuzzy_covariance_matrix+0.00001*np.random.rand(*fuzzy_covariance_matrix.shape)) @ (centroids[i]-xk))), 1, xks))
    if add_imaginary_center:
        d[:,-1] = 10 * np.trace(np.cov(xks, rowvar=False))
    memberships = d ** (1/(1-fuzziness))
    memberships = memberships/(np.sum(memberships, axis=1)[:, np.newaxis])
    # TODO: check if needed
    memberships[np.isnan(memberships)] = 1
    return memberships


def fuzzy_kmeans(xks, k, fuzziness, initial_centroids=None, initial_memberships=None, max_iter=500, seed=None, distance_type="euclidean",
                 eps=1e-5, fuzziness_decay=False, iters_to_decay=50, decay_factor=5,  add_imaginary_center=False):
    if xks.ndim == 1:
        xks = xks[:, np.newaxis]
    # random init
    centroids = kmeans(xks=xks, k=k, seed=seed)[0] if initial_centroids is None else initial_centroids[:]
    # inital memberships matrix
    n = xks.shape[0]
    k = centroids.shape[0]
    old_memberships = np.zeros((n, k)) if initial_memberships is None else initial_memberships
    for i in range(max_iter):
        if i == 0:
            memberships = find_memberships(xks, centroids, fuzziness, old_memberships, distance_type, add_imaginary_center)
            if add_imaginary_center:
                old_memberships = np.concatenate([old_memberships, np.zeros((n, 1))], axis=1) 
        else:
            memberships = find_memberships(xks, centroids, fuzziness, old_memberships, distance_type)
        centroids = find_centroids(memberships, xks, fuzziness)
        # check if there is coverage
        if np.abs(memberships - old_memberships).max() <= eps:
            print("converged after", i, "iterations")
            break
        else:
            old_memberships = memberships
        if fuzziness_decay and (i + 1) % iters_to_decay == 0 and fuzziness*np.exp(-decay_factor) >= 2:
            fuzziness = fuzziness*np.exp(-decay_factor)
            print("fuzziness was decaied to", fuzziness)
    return centroids, memberships, fuzziness