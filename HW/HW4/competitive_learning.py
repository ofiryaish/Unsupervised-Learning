import numpy as np
from sklearn.preprocessing import normalize


def training(xks, alpha, iterations, n_output, eps=1e-5, num_to_stop_without_change=100):
    # xks should be normalized
    xks = normalize(xks, norm="l2")
    # initiliztion
    alpha0 = alpha
    count_no_significant_change_in_w = 0
    dim = xks.shape[1] # date dim
    n_samples = len(xks) # total samples in training data
    # initialize weights
    W = np.random.rand(n_output, dim)
    W = normalize(W, norm="l2")

    # create a random number generator to sample the dataset
    rng = np.random.default_rng()

    for t in range(iterations):
        # learning rate decay
        alpha = alpha0*(0.99)**t   

        # random input
        rand_i = rng.integers(n_samples)
        input_vec = xks[rand_i]

        # find the winning neuron
        win_index = np.argmax(np.dot(W, input_vec))

        # step
        w_old = W[win_index].copy()
        W[win_index] += alpha * (input_vec - W[win_index])
        # or
        # W[win_index] += alpha * input_vec
        W[win_index] = normalize(W[win_index], norm="l2")

        if np.linalg.norm(w_old - W[win_index]) < eps: 
            count_no_significant_change_in_w += 1
        else:
            count_no_significant_change_in_w = 0
        if num_to_stop_without_change == count_no_significant_change_in_w:
            print("break after", t)
            break

    return W