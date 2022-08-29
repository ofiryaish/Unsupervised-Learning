import numpy as np
from HW.HW2.fuzzy_kmeans import fuzzy_covariance

def fuzzy_invariant_criterion(xks, memberships, centroids):
    """
    xks: nXm
    meberships: nxk
    centroids: kxm
    q: sclar
    """
    k = centroids.shape[0]
    # The scatter matrix is the fuzzy covariance
    S = np.array([fuzzy_covariance(memberships[:, i], centroids[i], xks) for i in range(k)])
    # within-cluster scatter matrix
    SW = np.sum(S, axis=0)
    # betwwen-cluster scatter matrix
    m = np.mean(xks, axis=0)
    cluster_i_sum_of_memberships = np.array([memberships[:, i].sum() for i in range(k)])[:, np.newaxis, np.newaxis] # non-normlized
    SB = np.sum(cluster_i_sum_of_memberships * ((centroids - m)[:,:, np.newaxis] @ (centroids - m)[:,np.newaxis, :]), axis=0)
    
    return np.trace(np.matmul(np.linalg.inv(SW), SB))


def invariant_criterion(xks, memberships):
    """
    xks: nXm
    meberships: nxk
    """
    # invariant to the number of clusters
    k = memberships.shape[1]
    y = np.argmax(memberships, axis=1)
    m = np.mean(xks, axis=0)

    SB = 0
    SW = 0

    for i in range(k):
        xks_i = xks[y == i]
        mi = np.mean(xks_i, axis=0)
        d = xks_i - mi
        Si = np.sum(d[:, :, None] @ d[:, None, :], axis=0) # scatter matrix
        SW += Si # within cluster scatter matrix
        d = mi - m
        d = d[:, None]
        SB += len(xks_i) * (d @ d.T) # between cluster scatter matrix

    return np.trace(np.matmul(np.linalg.inv(SW), SB))


def hypervolume_criterion(xks, centroids, memberships):
    #average cluster volume
    k = centroids.shape[0]
    fuzzy_covariance_matrices_det = np.array([np.linalg.det(fuzzy_covariance(memberships[:, i], centroids[i], xks)) for i in range(k)])
    fuzzy_covariance_matrices_det = fuzzy_covariance_matrices_det ** 0.5
    
    return 1/(fuzzy_covariance_matrices_det.sum())


def partition_density(xks, centroids, memberships):
    #density of the central members
    k = centroids.shape[0]
    h = 0
    c = 0

    for i in range(k):
        fuzzy_covariance_matrix = fuzzy_covariance(memberships[:, i], centroids[i], xks)
        h += np.linalg.det(fuzzy_covariance_matrix) ** 0.5
        d = xks - centroids[i]
        sv = (d @ np.linalg.inv(fuzzy_covariance_matrix)) * d #Mahanalobis distance between x and the cluster centers
        m_ind = np.argwhere((sv[:, 0] < 1) * (sv[:, 1] < 1)) #define central members
        
        c += np.sum(memberships[:, i][m_ind])

    return c / h


def average_partition_density(xks, centroids, memberships, method='APDC', alpha=0.5):
    #average patition density of the central/maximal members
    k = centroids.shape[0]
    ad = 0
    for i in range(k):
        fuzzy_covariance_matrix = fuzzy_covariance(memberships[:, i], centroids[i], xks)
        h = np.linalg.det(fuzzy_covariance_matrix) ** 0.5
        if method == 'APDC':
            d = xks - centroids[i]
            sv = (d @ np.linalg.inv(fuzzy_covariance_matrix)) * d
            m_ind = np.argwhere((sv[:, 0] < 1) * (sv[:, 1] < 1))
        elif method == 'APDM':
            m_ind = np.argwhere(memberships[:, i] == np.max(memberships[:, i]))

        if not len(m_ind):
            c = 0
        else:
            c = np.sum(memberships[:, i][m_ind])

        ad += c / h

    V = ad / k

    return V


def normalized_partition_indexes(xks, memberships, centroids, fuzziness):
    #normalized partition density
    # TODO: check if the distance should be L2-norm
    k = centroids.shape[0]
    d = 0
    for i in range(k):
        d += (memberships[:, i] ** fuzziness) * np.sum((xks - centroids[i]) ** 2, axis=1)

    return k * np.sum(d)


def validity_scores_fun(xks, memberships, centroids, method, fuzziness=None):
    if method == 'FINV':  # maximize
        return fuzzy_invariant_criterion(xks, memberships, centroids)
    elif method == 'INV':  # maximize
        return invariant_criterion(xks, memberships)
    elif method == '1/HV':  # maximize
        return hypervolume_criterion(xks, centroids, memberships)
    elif method == 'PD':  # maximize
        return partition_density(xks, centroids, memberships)
    elif (method == 'APDM') or (method == 'APDC'):  # maximize
        return average_partition_density(xks, centroids, memberships, method=method)
    elif method == '1/NPI':  # minimize therfore 1/NPI to maximize
        return 1/normalized_partition_indexes(xks, memberships, centroids, fuzziness)