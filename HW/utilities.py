import itertools
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredText


def unit_vector(vector):
    """
    Returns the unit vector of the vector

    Parameters
    ----------
    vector : numpy array
    
    Returns
    -------
    v : numpy array
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    Parameters
    ----------
    v1 : numpy array
    v2 : numpy array
    
    Returns
    -------
    n : float
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def plot_scatter(data, reduction_type="TSNE", outliers=False, file_name=None, title=None, sigmas=None, mus=None,
                 v_score=None, show=True):
    """
    creates a scatter plot of the data in 2D or 3D

    Parameters
    ----------
    data : List or any sequential structre that holds the cluster data. For example, [xks_1, xks_2, ...]
           where xks_i is a numpy array of n datapints of dimension m (n,m) of cluster i
    reduction_type : str that define the dimension reduction to apply to visulize 4 or more dimensinal data.
                     Options: "TSNE", "PCA", Default: "TSNE"
    outliers : Boolean that define whether the last element is data is an outliers set or not. 
               If True visulize the outliers differently from clusters. Releant for 2D. Default: False
    file_name : str or None that define the file name of the figure when saving.
               If None, does not save the figure. Default: None
    title : str or None that define the figure title. If not, no title. Default: None
    sigmas : None or List or any sequential structre that holds the clusters covariance matrices.
             Relevant to 2D gaussian mixture model. If None, does not show the ellipse learned.
             For example, [sigma_1, sigma_2, ...] where sigma_i is a numpy array (m, m) of the covariance of cluster i.
             Default: None
    mus : None or List or any sequential structre that holds the clusters means/centroids. Relevant to 2D data.
          Similar to sigmas with elements of shape (m,1) or (m,). if None, does not show the means on plot.
          Default: None
    v_score : float or None. V-measure score. if None, does not disply the measure on plot. Default: None
    show : Boolean that indicate whether to show the plot or return the plot object. Default: True

    Returns
    -------
    p : None or matplotlib Figure object
    """
    if outliers:
        outliers_data = data[-1]
        data = data[:-1]
    
    sets = len(data)
    dim = data[0].shape[1]
    # dimension reduction to 2D in case of 4D data or more
    if dim > 3:
        sigmas, mus = None, None
        xks = np.concatenate(data)
        sets_sizes = [len(data[i]) for i in range(sets)]
        if reduction_type == "TSNE":
            xks_trans = TSNE(n_components=2, learning_rate=200, init='pca', perplexity=10).fit_transform(xks)
        else:
            pca = PCA(n_components=2)
            pca.fit(xks)
            xks_trans = pca.transform(xks)
        data = [xks_trans[sum(sets_sizes[:i]):sum(sets_sizes[:i+1])] for i in range(sets)]

    dim = data[0].shape[1]
    fig = plt.figure()
    # 3D plot
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(sets):
            ax.scatter(data[i][ :, 0], data[i][ :, 1], data[i][ :, 2], label=f'datapoints={data[i].shape[0]}', s=2)
        if outliers:
            ax.scatter(outliers_data[ :, 0], outliers_data[ :, 1], outliers_data[ :, 2], label=f'outliers datapoints={outliers_data.shape[0]}', s=15, c="black")
        plt.grid(color='k', linestyle=':', linewidth=1)
    # 2D plot 
    elif dim == 2:
        ax = fig.add_subplot(111)
        for i in range(sets):
            ax.scatter(data[i][ :, 0], data[i][ :, 1], label=f'datapoints={data[i].shape[0]}', s=2)
            # plot ellipse if provided. based on the eigendecomposition 
            if sigmas is not None:     
                eigenvalues, eigenvectors = np.linalg.eig(sigmas[i])
                eigenvalues = np.sqrt(eigenvalues)
                print(eigenvectors[1,0])
                angle = np.rad2deg(angle_between(eigenvectors[0], np.array([1, 0])))
                
                angle = angle if eigenvectors[1,0] >= 0 else 360-angle
                ell = Ellipse(xy=(mus[i][0], mus[i][1]), width=eigenvalues[0] * 2, height=eigenvalues[1] * 2, fill=False,
                              angle=angle,  alpha=1)
                plt.gca().add_artist(ell)
            # plot centers if provided
            if mus is not None:
                ax.scatter(mus[i][0], mus[i][1], c="grey", marker="X")
        # plot outliers
        if outliers:
            ax.scatter(outliers_data[ :, 0], outliers_data[ :, 1], label=f'outliers datapoints={outliers_data.shape[0]}', s=30, c="black")

    if v_score is not None:
        anchored_text = AnchoredText("V-measure: {}".format(round(v_score, 3)), loc="lower center")
        ax.add_artist(anchored_text)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    if file_name is not None:
        plt.savefig('{}.png'.format(file_name), bbox_inches='tight')
    if show:
        plt.show()
    else:
        return fig



def plot_prob(class_sizes, prob_matrix, show=True):
    """
    creates a plot of the membership probilities as function of the sample points

    Parameters
    ----------
    class_sizes : List or any sequential structre that holds the clsustrs' sizes. 
    prob_matrix : numpy array (n, c) where n is the number of samples, and c is the number of clusters.
                  for each samaple, contains the membership probilities:
                  prob_matrix[i, j] = membership probility of sample i to cluster j.
                  Assumes that the first sum(class_sizes[:i]):sum(class_sizes[:i+1]) elements
                  of prob_matrix are the samples of class i
    show : Boolean that indicate whether to show the plot or return the plot object. Default: True
    
    Returns
    -------
    p : None or matplotlib Figure object
    """
    n = len(prob_matrix)
    c = len(class_sizes)
    for i in range(c):
        class_loc_in_prob_matrix_start = sum(class_sizes[:i])
        class_loc_in_prob_matrix_end = class_loc_in_prob_matrix_start + class_sizes[i]
        print(class_loc_in_prob_matrix_end)
        if i < c-1:
            plt.axvline(class_loc_in_prob_matrix_end, c="black")
        plt.plot(range(1, n+1), prob_matrix[:, i]+i, linewidth=2-i*0.2)
    plt.scatter(range(1, n+1), np.argmax(prob_matrix, axis=1), marker="X", s=10, c="red")

    if show:
        plt.show()
    else:
        return plt. gcf()
    
    
    
def plot_validitiy(validitiy_scores_dict, show=True):
    """
    creates a subplots the cluster validitiy scores as function of number of clusters
    supports up to nine scores on 3x3 subplots
    
    Parameters
    ----------
    validitiy_scores_dict : dictionary {k -> v_l} where k is the validitidy score and v_l is list 
                            or any sequential structre that contains the scores for this type. 
                            
    Returns
    -------
    p : None or matplotlib Figure object
    """

    fig, ax = plt.subplots(3, 3, figsize=(25, 5))
    ax = ax.flatten()

    for a, validity_score_key in itertools.zip_longest(ax, validitiy_scores_dict):
        if a.axes is None:
            continue
        if validity_score_key is None:
            fig.delaxes(a) #The indexing is zero-based here
            continue
        validitiy_scores = validitiy_scores_dict[validity_score_key]
        a.plot(range(1, len(validitiy_scores) + 1), validitiy_scores_dict[validity_score_key])
        a.set_title(validity_score_key, fontsize=12)
        a.set_xlabel('Number of clusters', fontsize=8)
        a.set_xticks(range(1, len(validitiy_scores) + 1))

        a.tick_params(axis='both', which='major', labelsize=8)
        a.tick_params(axis='both', which='minor', labelsize=8)
        
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return plt. gcf()