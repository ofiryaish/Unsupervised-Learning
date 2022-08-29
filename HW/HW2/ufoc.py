import numpy as np

from HW.HW2.fuzzy_kmeans import fuzzy_kmeans
from HW.HW2.cluster_validity import validity_scores_fun


def uofc(xks, k_max, fuzziness=3, max_iter=500, seed=None,
         eps=1e-5, fuzziness_decay=False, iters_to_decay=50, decay_factor=5):
    validity_scores = {"1/HV": [], "PD": [], "APDC": [], "APDM": [],
                       "FINV": [], "INV": [], "1/NPI":[]}
    centroids_old, memberships_old = [], []
    n = xks.shape[0]
    centroids = np.mean(xks, axis=0)[np.newaxis, :]
    memberships = np.zeros((n, 1))
    for k in range(1, k_max + 1):
        print("k =", k)
        # memberships = np.zeros(memberships.shape)
        # regular fuzzy K-means
        # TODO: check if there is need to add imaginary center already in the first interation of the fuzzy K-means
        centroids, memberships, updated_fuzziness = fuzzy_kmeans(xks, k=k, fuzziness=fuzziness, initial_centroids=centroids,
                                                                initial_memberships=memberships, max_iter=max_iter,
                                                                seed=seed, distance_type="euclidean", eps=eps,
                                                                fuzziness_decay=False, iters_to_decay=iters_to_decay, decay_factor=decay_factor,
                                                                add_imaginary_center=True if k > 1 else False)
        # MLE fuzzy K-means
        centroids, memberships, updated_fuzziness = fuzzy_kmeans(xks, k=k, fuzziness=fuzziness, initial_centroids=centroids,
                                                                 initial_memberships=memberships, max_iter=max_iter,
                                                                 seed=seed, distance_type="exponential", eps=eps,
                                                                 fuzziness_decay=fuzziness_decay, iters_to_decay=iters_to_decay, decay_factor=decay_factor,
                                                                 add_imaginary_center=False)

        for validity_method in validity_scores:
            validity_scores[validity_method].append(validity_scores_fun(xks, memberships, centroids, method=validity_method, fuzziness=fuzziness))
            
        centroids_old.append(centroids)
        memberships_old.append(memberships)
    return centroids_old, memberships_old, validity_scores
