import numpy as np
import bct
from IPython import embed


# def clustering_coef_wu(weighted_edges):
#     """
#     CLUSTERING_COEF_WU     Clustering coefficient

#     C = clustering_coef_wu(W)

#     The weighted clustering coefficient is the average "intensity"
#     (geometric mean) of all triangles associated with each node.

#     Input:      W,      weighted undirected connection matrix
#                         (all weights must be between 0 and 1)

#     Output:     C,      clustering coefficient vector

#     Note:   All weights must be between 0 and 1.
#             This may be achieved using the weight_conversion.m function,
#             W_nrm = weight_conversion(W, 'normalize')

#     Reference: Onnela et al. (2005) Phys Rev E 71:065103


#     Mika Rubinov, UNSW/U Cambridge, 2007-2015

#     Modification history:
#     2007: original
#     2015: expanded documentation
#     """
#     K = np.sum(weighted_edges != 0, axis=1).astype(float)
#     cyc3 = weighted_edges ** (1 / 3.)
#     cyc3 = np.diag(np.linalg.matrix_power(cyc3, 3))
#     K[cyc3 == 0] = np.inf  # if no 3-cycles exist, make C=0 (via K=inf)
#     C = cyc3 / (K * (K-1))  # clustering coefficient
#     return C


def find_grid_communities(C):
    """FCN_GRID_COMMUNITIES       outline communities along diagonal

       [X Y INDSORT] = FCN_GRID_COMMUNITIES(C) returns {X,Y} coordinates for
                       highlighting modules located along the diagonal.
                       INDSORT are the indices to sort the matrix.

       Inputs:     C,       community assignments

       Outputs:    X,       x coor
                   Y,       y coor
                   INDSORT, indices

       Richard Betzel, Indiana University, 2012
    """
    C = np.sort(C)
    indsort = np.argsort(C)

    X = []
    Y = []
    for i in np.unique(C):
        ind = np.where(C == i)[0]
        if len(ind) > 0:
            mn = np.min(ind) - 0.5
            mx = np.max(ind) + 0.5
            X.extend([mn, mn, mx, mx, mn, np.nan])
            Y.extend([mn, mx, mx, mn, mn, np.nan])
    return X, Y, indsort


def order_partition(A, communities, random=False):
    """
    A: Adjacency Matrix
    communities: community assignment vector
    random: Shuffle nodes within each community
    """

    # n_communities = np.max(communities)
    n_nodes = len(communities)
    # counts, _ = np.histogram(communities, range(n_communities))

    # [~,indsort] = sort(h,'descend')
    ind_sorted = np.zeros(n_nodes, dtype=int)
    ci_sorted = np.zeros(n_nodes, dtype=int)
    count = 0
    for c in np.unique(communities):
        c_ind = communities == c

        y = A[c_ind, :][:, c_ind]
        within_weights = np.mean(y, 1) + np.mean(y, 0).T

        yy = A[c_ind, :][:, ~c_ind]
        yy2 = A[~c_ind, :][:, c_ind]
        between_weights = np.mean(yy, 1) + np.mean(yy2, 0).T

        z = within_weights - between_weights
        vtx = np.where(c_ind)[0]
        jnd = np.argsort(z)[::-1]
        if random is True:
            jnd = np.random.choice(jnd, replace=False)
        vtx = vtx[jnd]

        ind_sorted[count: count + len(jnd)] = vtx
        ci_sorted[count: count + len(jnd)] = c
        count += len(jnd)
    return ind_sorted, ci_sorted


def fcn_plot_blocks(g):
    """FCN_PLOT_BLOCKS    visualize block structure

      [X,Y,IND] = FCN_PLOT_BLOCKS is a visualization tool for plotting block
                  model boundaries. As input it takes a vec G of size N x 1
                  (N is the number of nodes) where each cell is the group
                  assignment for the corresponding node. As an output, it
                  returns X and Y, which are vectors of boundaries and also
                  IND, which is the ordering of the adjacency matrix to fit
                  with the block structure.

      Example:    >> [partioned,q] = modularity_louvain_und(A) # find modules
                  >> [x,y,ind] = fcn_plot_blocks(partioned)    # get vectors
                  >> imagesc(A(ind,ind)) hold on plot(x,y,'w')

      Inputs:     G,      vector of group assignments

      Outputs:    X,      x-coordinates
                  Y,      y-coordinates
                  IND,    reordering for adjacency matrix

      Richard Betzel, Indiana University, 2013
    """
    nc = max(g)
    n = length(g)
    x = 0.5
    y = n + 0.5
    [gsort, ind] = np.sort(g)
    X = []
    Y = []
    for i in range(2, nc):
        aa = np.where(gsort == i)
        xx = [x, y]
        yy = (aa[0] - 0.5) * np.ones(2, 1)
        X = [X, xx, np.nan, yy, np.nan]
        Y = [Y, yy, np.nan, xx, np.nan]
    return X, Y, ind


def edges_to_modularity_matrix(weighted_edges, resolution=1.):
    """
    Parameters
    ----------
    weighted_edges : array, shape (n_nodes, n_nodes)
        Connectivity matrix for a graph.

    Returns
    -------
    modularity_matrix : array, shape (n_nodes, n_nodes)
        Modularity matrix for this connectivity matrix.
    """
    edge_sum = float(weighted_edges.sum().sum())
    P = np.outer(np.sum(weighted_edges, 1), np.sum(weighted_edges, 1))
    P /= np.sum(weighted_edges)
    modularity_matrix = (weighted_edges - resolution * P) / edge_sum
    return modularity_matrix


def genlouvain_over_time(graphs, gamma=1., omega=1.):
    """Calculate genlouvain partition over time

    Parameters
    ----------
    graphs : array, shape (n_nodes, n_nodes, n_times)

    Returns
    -------
    partition_time : array, shape(n_nodes, n_times)
    q_time : float
    """
    n_nodes, _, n_time = graphs.shape
    conn_matrix = np.zeros([n_nodes * n_time, n_nodes * n_time])
    twomu = 0
    for t in range(n_time):
        this_mat = np.squeeze(graphs[..., t])

        k = np.sum(this_mat, axis=0)
        twom = np.sum(k)
        twomu = twomu + twom

        indx = np.arange(n_nodes) + t * n_nodes
        iinds, jinds = np.meshgrid(indx, indx)
        conn_matrix[iinds, jinds] = this_mat - gamma * np.outer(k, k) / twom

    twomu = twomu + 2 * omega * n_nodes * (n_time - 1)
    n_rows = n_nodes * n_time
    diags = np.diag(np.ones(n_rows), -n_nodes) + np.diag(np.ones(n_rows), n_nodes)
    diags = diags[:n_nodes * n_time, :n_nodes * n_time]
    conn_matrix = conn_matrix + omega * diags

    # Generate the louvain partition
    partition_time, q_time = bct.community_louvain(np.ones_like(conn_matrix),
                                                   gamma=gamma, B=conn_matrix)
    q_time = q_time / twomu

    # Reshape so time ix X axis
    partition_multi = partition_time.reshape(n_time, n_nodes).T
    return partition_multi, q_time
