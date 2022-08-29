import numpy as np
from tqdm import tqdm

from sklearn.metrics import pairwise_distances

###################################
############distances##############
###################################
def find_nearest(xks_1, xks_2):
    xks_1, xks_2 = np.asarray(xks_1), np.asarray(xks_2)
    # equvlant to pairwise_distances(xks_1, xks_2, metric="euclidean")
    d_matrix = np.sqrt(np.sum((xks_1[:, np.newaxis,:] - xks_2[np.newaxis, :, :]) ** 2, axis=2))
    return np.min(d_matrix)


def find_farthest(xks_1, xks_2):
    xks_1, xks_2 = np.asarray(xks_1), np.asarray(xks_2)
    d_matrix = np.sqrt(np.sum((xks_1[:, np.newaxis,:] - xks_2[np.newaxis, :, :]) ** 2, axis=2))
    return np.max(d_matrix)


def find_average(xks_1, xks_2):
    xks_1, xks_2 = np.asarray(xks_1), np.asarray(xks_2)
    d_matrix = np.sqrt(np.sum((xks_1[:, np.newaxis,:] - xks_2[np.newaxis, :, :]) ** 2, axis=2))
    return np.mean(d_matrix)


def means_dist(xks_1, xks_2):
    xks_1, xks_2 = np.asarray(xks_1), np.asarray(xks_2)
    return np.linalg.norm(xks_1.mean(axis=0) - xks_2.mean(axis=0))


def weighted_means_dist(xks_1, xks_2):
    xks_1, xks_2 = np.asarray(xks_1), np.asarray(xks_2)
    n1, n2 = xks_1.shape[0], xks_2.shape[0]
    return np.sqrt(n1*n2/(n1+n2)) * means_dist(xks_1, xks_2)


def set_ditance_fun(dist_func):
    if dist_func == "min":
        dist_func = find_nearest
    elif dist_func == "max":
        dist_func = find_farthest
    elif dist_func == "avg":
        dist_func = find_average
    elif dist_func == "mean":
        dist_func = means_dist
    elif dist_func == "weighted_mean":
        dist_func = weighted_means_dist

    return dist_func

###################################
############algoritm###############
###################################
def agglomerative_hirarchial_clustering(xks, dist_func, k=None):
    """
    if k is not none, then stop the hirachical clusring when there are k clusters, and return the clusters.
    if k is none, perform all process and return Z - the linkage matrix
    """
    dist_func = set_ditance_fun(dist_func)
    n, m = xks.shape
    # [cluster_id, [elememts in cluster], [indices in original xks of the elements]
    clusters = [[cluster_id, [xks[cluster_id]], [cluster_id]] for cluster_id in range(n)]
    # linkage matrix
    Z = np.zeros((n-1, 4))
    
    # Distance matrix
    D = pairwise_distances(xks, metric="euclidean")
    ind = np.tril_indices(n)
    D[ind] = np.inf

    for join_index in tqdm(range(n-1)):
        min_row, min_col = np.unravel_index(np.argmin(D), D.shape)
        cluser_loc_to_expand, cluser_loc_to_remove = min(min_row, min_col), max(min_row, min_col)
        clustr_id_to_expand, clustr_id_to_remove = clusters[cluser_loc_to_expand][0], clusters[cluser_loc_to_remove][0]
        
        # combine the cluster into cluser_loc_to_expand
        clusters[cluser_loc_to_expand][0] = join_index + n
        clusters[cluser_loc_to_expand][1].extend(clusters[cluser_loc_to_remove][1])
        clusters[cluser_loc_to_expand][2].extend(clusters[cluser_loc_to_remove][2])
        del clusters[cluser_loc_to_remove]

        # update the linkage matrix
        Z[join_index, 0] = clustr_id_to_expand
        Z[join_index, 1] = clustr_id_to_remove
        Z[join_index, 2] = np.min(D)
        Z[join_index, 3] = len(clusters[cluser_loc_to_expand][1])
        
        # delete the min_col cluster from the distance matrix
        D = np.delete(D, (cluser_loc_to_remove), axis=0)
        D = np.delete(D, (cluser_loc_to_remove), axis=1)
        
        # recompute the distances from the new cluser_loc_to_expand cluster
        for i in range(D.shape[0]):
            if cluser_loc_to_expand < i:
                D[cluser_loc_to_expand, i] = dist_func(clusters[cluser_loc_to_expand][1], clusters[i][1])
            elif i < cluser_loc_to_expand:
                D[i, cluser_loc_to_expand] = dist_func(clusters[cluser_loc_to_expand][1], clusters[i][1])
                
        if k is not None and len(clusters) == k:
            return clusters
                
    return Z