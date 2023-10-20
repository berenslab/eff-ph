import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
import scipy.linalg
from umap.umap_ import fuzzy_simplicial_set, UMAP
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.nearest_neighbors import KNNIndex
import os
import pickle
from sklearn.decomposition import PCA
from vis_utils.utils import load_dict, save_dict, kNN_dists, kNN_graph
from vis_utils.tsne_wrapper import TSNEwrapper


def sim_to_dense(dissim):
    """
    Converts a sparse dissimilarity matrix to a dense dissimilarity matrix. All values are shifted so that they are
    non-negative. Non-explicit zeros are set to twice the maximal value. The diagonal is set to the minimal value.
    :param dissim: sparse coo matrix
    :return: dense array
    """
    dissim.data -= dissim.data.min()  # sets the lowest dissimilarity to zero
    dense_dissim = dissim.toarray()

    max_dissim = dense_dissim.max()

    # sets all zeros to large value (both explicit and non-explicit)
    dense_dissim[dense_dissim == 0] = 2 * max_dissim

    # restores the explicit zeros to their value
    expl_zeros = dissim.data == 0.0
    dense_dissim[dissim.row[expl_zeros], dissim.col[expl_zeros]] = 0.0

    dense_dissim[np.eye(len(dense_dissim)).astype(bool)] = 0.0

    return dense_dissim


def get_core_dist(x, k, input_distance="euclidean"):
    """
    get the core distance, that underlies HDBSCAN. Very similar to DTM when it uses p_radius, p_dtm = np.inf
    :param x: data
    :param k: number of nearest neighbors
    :param input_distance: input distance based on which to compute this distance. Must be one of ["euclidean", "cosine", "correlation]
    :return: pairwise distance matrix of core distances
    """
    d_eucl = squareform(pdist(x, metric=input_distance))
    knn_dists = kNN_dists(x, k=k, metric=input_distance).cpu().numpy()
    core = knn_dists.max(axis=1)
    core_x, core_y = np.meshgrid(core, core)
    d_core = np.stack([d_eucl, core_x, core_y], axis=0).max(0)
    d_core[np.eye(len(d_core), dtype=bool)] = 0
    return d_core


def get_sknn(x, k=15, metric="euclidean"):
    """
    get unweighted sknn graph
    :param x: data
    :param k: number of nearest neighbors
    :param metric: metric for kNN, must be one of ["euclidean", "cosine", "correlation]
    :return: sparse coo matrix of unweighted knn sknn graph
    """
    knn_graph = kNN_graph(x.astype("float"),
                          k,
                          metric=metric).cpu().numpy().flatten()
    knn_graph = sp.coo_matrix((np.ones(len(x) * k),
                                         (np.repeat(np.arange(x.shape[0]), k),
                                          knn_graph)),
                                        shape=(len(x), len(x)))
    sknn_graph = knn_graph.maximum(knn_graph.transpose()).tocoo()
    return sknn_graph


def get_distance_weighted_sknn(x, k=15, metric="euclidean"):
    """
    get distance weighted sknn graph
    :param x: data
    :param k: number of nearest neighbors
    :param metric: metric for kNN, must be one of ["euclidean", "cosine", "correlation]
    :return: sparse coo matrix of distance weighted sknn graph
    """
    knn_graph = kNN_graph(x, k=k, metric=metric).cpu().numpy().flatten()

    knn_dist = kNN_dists(x, k=k, metric=metric).cpu().numpy()
    knn_dist_coo = sp.coo_matrix((knn_dist.flatten(), (np.repeat(np.arange(len(x)), k), knn_graph)),
                                           shape=(len(x), len(x)))

    sknn_dist_coo = knn_dist_coo.maximum(knn_dist_coo.transpose()).tocoo()
    return sknn_dist_coo


def get_sp_sknn(x, k=15, weighted=True, input_distance="euclidean"):
    """
    get the shortest path distane on (un)weighted sknn graph
    :param x: data
    :param k: number of nearest neighbors
    :param weighted: if True, use distance weighted graph
    :return: shortest path distance matrix
    """
    if weighted:
        sknn_dist_coo = get_distance_weighted_sknn(x, k=k, metric=input_distance)
    else:
        sknn_dist_coo = get_sknn(x, k=k, metric=input_distance)

    sp_dists = sp.csgraph.shortest_path(sknn_dist_coo)
    # handle disconnected components
    sp_dists[sp_dists == np.inf] = sp_dists[sp_dists != np.inf].max() * 2
    return sp_dists


def get_dtm_weights(x, k, p_dtm=np.inf, p_radius=np.inf, input_distance="euclidean"):
    """
    get the weights for the DTM-based dissimilarities
    :param x: data
    :param k: number of nearest neighbors
    :param p_dtm: power that integrates the distances to nearest neighbors into the distance to measure
    :param p_radius: controls the aggregation of the distance-to-measure and the input distance to the dissimilarity
    :param input_distance: input distance based on which to compute this distance. Must be one of ["euclidean", "cosine", "correlation]
    :return: dtm dissimilarity matrix
    """
    # get distance to measure for each point
    dtm = get_dtm(x, k, p=p_dtm, input_distance=input_distance)

    # compute input distance
    d = squareform(pdist(x, metric=input_distance))

    # mix dtm with input distances
    dtm_x, dtm_y = np.meshgrid(dtm, dtm)

    # criterion for using the max dtm or the aggregated value
    if np.isinf(p_radius):
        dtm_diff = np.maximum(dtm_x, dtm_y)
    else:
        dtm_diff = np.abs(dtm_x**p_radius - dtm_y**p_radius) ** (1 / p_radius)

    mask_singleton = d <= dtm_diff

    # compute aggregated value
    if p_radius == 1:
        mixed_filt_val = (dtm_x + dtm_y + d) / 2
    elif p_radius == 2:
        mixed_filt_val = np.sqrt(((dtm_x + dtm_y)**2 + d**2) * ((dtm_x - dtm_y)**2 + d**2)) \
                          / (2 * d + np.eye(len(d)) + 1e-10)
        # addition of identity matrix to avoid division by zero
    elif p_radius == np.inf:
        mixed_filt_val = np.stack([dtm_x, dtm_y, d/2], axis=0).max(0)
    else:
        raise ValueError("p must be 1, 2 or np.inf")

    # return max dtm or aggregated value based on dtm_diff criterion
    return np.maximum(dtm_x, dtm_y) * mask_singleton + mixed_filt_val * ~mask_singleton


def get_dtm(x, k, p=np.inf, input_distance="euclidean"):
    """
    get distance to measure graph
    :param x: data
    :param k: number of nearest neighbors
    :param p: power of distances
    :param input_distance: input distance based on which to compute this distance. Must be one of ["euclidean", "cosine", "correlation]
    :return: pairwise distance matrix
    """
    knn_dists = kNN_dists(x, k=k, metric=input_distance).cpu().numpy()

    # cannot use np.linalg.norm due to factor 1/k in the root of the mean
    if p < np.inf:
        dtm = (1/k * (knn_dists**p).sum(1))**(1/p)
    else:
        dtm = knn_dists.max(axis=1)
    return dtm


def get_fermat_dist(x, p, input_distance):
    """
    computes fermat distance
    :param x: data
    :param p: power of distances
    :param input_distance: input distance based on which to compute this distance. Must be one of ["euclidean", "cosine", "correlation]
    :return: pairwise distance matrix
    """
    d_input = squareform(pdist(x, metric=input_distance))
    return sp.csgraph.shortest_path(d_input**p, directed=False)


def compute_laplacian(A, normalization="none"):
    """
    Computes the Laplacian based ona an adjacency matrix given as np.ndarray or scipy.sparse matrix. Based on code by
    Enrique Fita Sanmartin.

    :param A: adjacency matrix given as np.ndarray or scipy.sparse matrix
    :param normalization: whether to use no normalization ("none"), random walk normalization ("rw") or symmetric normalization ("sym")
    :return: Laplacian matrix in the same format as A
    """
    # compute degree matrix
    degs = A.sum(0)
    if isinstance(A, np.ndarray):
        D = np.diag(degs.flatten())
    else:
        D = sp.diags(np.asarray(degs).reshape(-1), format="csc")

    # compute non-normalized Laplacian
    L = D - A

    if normalization != "none":
        assert degs.min() > 0, "Graph contains nodes with zero degree."

        if normalization == "rw":
            if isinstance(A, np.ndarray):
                D_inv = np.diag(degs.flatten() ** (-1))
            else:
                D_inv = sp.diags(np.asarray(degs).reshape(-1) ** (-1), format="csc")
            L = D_inv @ L
        elif normalization == "sym":
            if isinstance(A, np.ndarray):
                D_inv_sqrt = np.diag(degs.flatten() ** (-0.5))
            else:
                D_inv_sqrt = sp.diags(np.asarray(degs).reshape(-1) ** (-0.5), format="csc")
            L = D_inv_sqrt @ L @ D_inv_sqrt
        else:
            raise NotImplementedError
    return L


def compute_effective_resistance_connected(A):
    """
    Computes the effective resistance using the pseudoinverse of the Laplacian L^+ of a connected graph.

    EffR[i,j]=L^+[i,i]+L^+[j,j]-2*L^+[i,j]

    Based on code by Enrique Fita Sanmartin.

    :param A: adjacency matrix (numpy or scipy.sparse array)
    :return: all pairs of effective resistance distances (numpy array)
    """

    n = A.shape[0]
    L = compute_laplacian(A)
    if not isinstance(L, np.ndarray):
        L = L.A

    Lpinv = np.linalg.inv(L + np.ones(L.shape) / n) - np.ones(L.shape) / n

    Linv_diag = np.diag(Lpinv).reshape((n, 1))
    EffR = Linv_diag * np.ones((1, n)) + np.ones((n, 1)) * Linv_diag.T - 2 * Lpinv

    return EffR


def compute_effective_resistance(A, disconnect=False):
    """
    Computes the effective resistance using the pseudoinverse of the Laplacian L^+ of an arbitrary graph. We will compute
    the effective resistance on each component separately and set the resistance between different components to inf.
    :param A: Adjacency matrix (np.ndarry or scipy.sparce matrix)
    :param disconnect: whether to compute the effective resistance for each connected component separately
    :return: all pairs of effective resistance distances (np.ndarray)
    """
    if disconnect:
        # compute connected components
        n_components, component_labels = sp.csgraph.connected_components(A)
        EffR = np.ones(A.shape) * np.inf  # initialize EffR matrix to inf for correct value between connected components

        for i in range(n_components):
            component_mask = component_labels == i
            component = np.where(component_mask)[0]
            component_mask = component_mask[:, None] @ component_mask[None, :]
            if sp.issparse(A):
                A = A.tocsr()
            # compute effective resistance on component
            EffR_component = compute_effective_resistance_connected(
                A[component, :][:, component])  # funny slicing for sparse matrices
            EffR[component_mask] = EffR_component.flatten()


    else:
        EffR = compute_effective_resistance_connected(A)

    # replace infinite values with twice the maximal finite value
    max_EffR = np.max(EffR[np.isfinite(EffR)])
    EffR[np.isinf(EffR)] = max_EffR * 2
    return EffR


def get_eff_res(x, k, corrected=True, weighted=False, disconnect=False, input_distance="euclidean"):
    """
    computes effective resistence distance on sknn graph
    :param x: data
    :param k: number of nearest neighbors
    :param corrected: whether to do the von Luxburg correction
    :param weighted: whether to use the weighted or unweighted knn graph
    :param disconnect: whether to compute the effective resistance on each connected component separately
    :param input_distance: input distance based on which to compute this distance. Must be one of ["euclidean", "cosine", "correlation]
    :return: pairwise distance matrix
    """

    # compute symmetric kNN graph
    if weighted:
        sknn_coo = get_distance_weighted_sknn(x, k=k, metric=input_distance)
    else:
        sknn_coo = get_sknn(x, k=k, metric=input_distance)

    # invert as edge weights are reciprocal of resistance
    sknn_coo.data = 1 / sknn_coo.data

    # compute effective resistance
    d_eff = compute_effective_resistance(sknn_coo, disconnect=disconnect)

    # optionally: correct via von Luxburg fix
    if corrected:
        degs = np.array(sknn_coo.sum(axis=1))
        deg_dist = 1/degs + 1/degs.T
        np.fill_diagonal(deg_dist, 0)
        d_eff = d_eff - deg_dist + 2*sknn_coo.toarray() / (degs * degs.T)
    return d_eff


def get_spectral_eff_res(x, k=15, corrected=False, weighted=False):
    """
    Computes the effective resistance distance based on the spectral decomposition of the Laplacian
    :param x: data (n_samples, n_features)
    :param k: number of nearest neighbors
    :param corrected: whether to do the von Luxburg correction
    :param weighted: whether to use the weighted or unweighted knn graph
    :return: effective resistance distance matrix
    """

    # compute symmetric kNN graph
    if weighted:
        sknn_coo = get_distance_weighted_sknn(x, k=k)
    else:
        sknn_coo = get_sknn(x, k=k)

    # invert as edge weights are reciprocal of resistance
    sknn_coo.data = 1 / sknn_coo.data

    # compute Laplacian, depending on whether we want to do the von Luxburg correction, we need a different
    # normalization
    if corrected:
        L = compute_laplacian(sknn_coo, normalization="sym")
    else:
        L = compute_laplacian(sknn_coo, normalization="none")

    eigenvalues, eigenvectors = scipy.linalg.eigh(
                    L.toarray(),
                )

    order = np.argsort(eigenvalues)[1:]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = eigenvalues[order]

    # Compute nD embedding. The decay is the main difference between the corrected and uncorrected version
    if corrected:
        decay = np.diag((1-eigenvalues)/np.sqrt(eigenvalues))
        D_sqrt_inv = np.diag(sknn_coo.sum(axis=0).A.flatten()**(-1/2))

        embd = D_sqrt_inv @ eigenvectors @ decay
    else:
        decay = np.diag(1/np.sqrt(eigenvalues))
        embd = eigenvectors @ decay

    # compute pairwise distances in embedding, effective resistance is the squqre of this
    dist = squareform(pdist(embd))
    return dist**2


def get_deg_dist(x, k, weighted=False, input_distance="euclidean"):
    """
    computes degree distance on sknn graph. This is not an informative metric, but just needed for the correction of the
     effective resistence distance
    :param x: data
    :param k: number of nearest neighbors
    :param weighted: whether to use the weighted or unweighted knn graph
    :param input_distance: input distance based on which to compute this distance. Must be one of
    ["euclidean", "cosine", "correlation]
    :return: pairwise distance matrix
    """
    # compute symmetric kNN graph
    if weighted:
        sknn_coo = get_distance_weighted_sknn(x, k=k, metric=input_distance)
    else:
        sknn_coo = get_sknn(x, k=k, metric=input_distance)

    # compute the degree distance
    degs = np.array(sknn_coo.sum(axis=1))
    deg_dist = 1/degs + 1/degs.T
    np.fill_diagonal(deg_dist, 0)
    return deg_dist


def get_spectral_dist(x,
                      k=15,
                      weighted=True,
                      n_evecs=2,
                      normalization="none",
                      return_embd=False,
                      input_distance="euclidean"):
    """
    Computes the spectral distance based on the spectral decomposition of the Laplacian. The is just the distance in the
     Laplacian Eigenmaps embedding of the data. Note that we exclude the first K eigenvectors if the skNN graph has K
     connected components.
    :param x: data (n_samples, n_features)
    :param k: number of nearest neighbors
    :param weighted: whether to use the weighted or unweighted knn graph
    :param n_evecs: number of eigenvectors to use for the embedding
    :param normalization: normalization of the Laplacian. Must be one of ["none", "sym", "rw"]
    :param return_embd: whether to return the embedding
    :param input_distance: input distance based on which to compute this distance. Must be one of
     ["euclidean", "cosine", "correlation]
    :return: distance matrix based on the Laplacian Eigenmaps embedding
    """

    # comptue symmetric kNN graph
    if weighted:
        sknn_coo = get_distance_weighted_sknn(x, k=k, metric=input_distance)
    else:
        sknn_coo = get_sknn(x, k=k, metric=input_distance)

    # compute number of connected components, so as to select the non-trivial eigenvalues
    n_components, component_labels = sp.csgraph.connected_components(sknn_coo)

    # compute Laplacian
    L = compute_laplacian(sknn_coo, normalization=normalization)

    # compute eigenvectors, following the setting in the UMAP code base
    num_lanczos_vectors = max(2 * (n_evecs+n_components) + 1, int(np.sqrt(L.shape[0])))
    eigenvalues, eigenvectors = sp.linalg.eigsh(
                    L,
                    n_evecs+n_components,
                    which="SM",
                    ncv=num_lanczos_vectors,
                    tol=1e-4,
                    v0=np.ones(L.shape[0]),
                    maxiter=L.shape[0] * 5,
                )

    # select the non-trivial eigenvectors and compute distance matrix
    order = np.argsort(eigenvalues)[n_components:n_evecs+n_components]
    dist = squareform(pdist(eigenvectors[:, order]))

    if return_embd:
        return dist, eigenvectors[:, order]
    return dist


def get_diffusion_dist(x, k=15, t=8, kernel="sknn", include_self=True, input_distance="euclidean"):
    """
    computes diffusion distance on sknn graph
    :param x: data
    :param k: number of nearest neighbors
    :param t: diffusion time
    :param kernel: kernel to use, must be one of "sknn", "gaussian"
    :return: pairwise distance matrix
    """
    # compute the graph on which we diffuse
    if kernel == "sknn":
        A = get_sknn(x, k=k, metric=input_distance)
    else:
        raise NotImplementedError("only sknn kernel implemented so far for diffusion distance")

    # add diagonal for self-loops in the diffusion
    if include_self:
        A = A + sp.eye(A.shape[0])

    # compute the diffusing transition matrix
    degrees = np.asarray(A.sum(1)).flatten()
    D_inv = sp.diags(degrees**(-1), format="csr")
    P = D_inv @ A
    P = P.toarray()

    # power it for t steps
    P_t = np.linalg.matrix_power(P, t)

    # The line below is the correct definition. The difference is only the last factor, which is a constant. Keeping
    # this version for legacy reasons.
    return squareform(pdist(P_t @ D_inv.toarray())) * np.sqrt(D_inv.sum())
    # return squareform(pdist(P_t @ D_inv.toarray())) * np.sqrt(degrees.sum()) # correct definition


def get_umap_dist(x, k, input_distance="euclidean", sim_to_dist="neg_log", use_rho=False, include_self=False):
    """
    Computes UMAP graph distance à la Gardner et al.
    :param x: data
    :param k: number of nearest neighbors
    :param input_distance: metric to use for kNN graph, must be one of "euclidean", "cosine", "correlation"
    :param sim_to_dist: how to transform the similarity to a distance, must be one of "neg_log", "neg"
    :param use_rho: whether to use the rhos from umap
    :param include_self: whether to include self in kNN graph
    :return: pairwise distance matrix
    """

    # compute kNN graph
    knn_graph = kNN_graph(x, k=k, metric=input_distance).cpu().numpy()
    knn_dists = kNN_dists(x, k=k, metric=input_distance).cpu().numpy()

    # add node itself to kNN graph, as usual in UMAP
    if include_self:
        knn_graph = np.concatenate([np.arange(len(x))[:, None], knn_graph], axis=1)
        knn_dists = np.concatenate([np.zeros(len(x))[:, None], knn_dists], axis=1)
        k_true = k + 1
    else:
        k_true = k

    # setting near zero distances to zero to handle rounding errors in the presence of duplicate inputs
    knn_dists[np.abs(knn_dists) < 1e-10] = 0.0

    # compute sigmas and rhos, do not use the similarities themselves for more control and better comparability between
    # use_rho=True and use_rho=False
    _, sigmas, rhos = fuzzy_simplicial_set(x,
                                           n_neighbors=k,  # should not matter if kkn graph is given
                                           random_state=42,  # should not matter if kkn graph is given
                                           metric='euclidean',  # should not matter if kkn graph is given
                                           verbose=True,
                                           knn_indices=knn_graph,
                                           knn_dists=knn_dists)


    # turn knn dists into sparse matrix
    if use_rho:
        sims = np.exp(- np.maximum(knn_dists - rhos[:, None], 0) / sigmas[:, None])
    else:
        sims = np.exp(- knn_dists / sigmas[:, None])

    sims = sp.coo_matrix((sims.flatten(),
                                    (np.repeat(np.arange(x.shape[0]), k_true),
                                     knn_graph.flatten())),
                                   shape=(len(x), len(x)))
    # symmetrize
    sims = sims + sims.T - sims.multiply(sims.T)
    sims = sims.tocoo()

    # due to numerical errors, some similarities might be non-positive. Remove those from the similarities
    pos_sim_mask = sims.data > 0
    sims.data = sims.data[pos_sim_mask]
    sims.row = sims.row[pos_sim_mask]
    sims.col = sims.col[pos_sim_mask]

    # transform similarities to dissimilarities
    if sim_to_dist == "neg_log":
        sims.data = -np.log(sims.data)
        dists = sim_to_dense(sims)
    elif sim_to_dist == "neg":
        sims.data = - sims.data
        dists = sim_to_dense(sims)
    else:
        raise NotImplementedError("sim_to_dist must be neg_log or neg")

    return dists


def get_umap_embd_dist(x,
                       k,
                       root_path,
                       dataset,
                       n_epochs=750,
                       seed=0,
                       input_distance="euclidean",
                       metric=None,
                       min_dist=0.1,
                       force_recompute=False):
    """
    Computes the distance between the UMAP embedding points.
    :param x: data
    :param k: number of nearest neighbors
    :param root_path: path to the root folder of the project
    :param dataset: name of the dataset
    :param n_epochs: number of epochs for UMAP
    :param seed: random seed for UMAP
    :param input_distance: metric to use for kNN graph, must be one of "euclidean", "cosine", "correlation"
    :param metric: deprecated, use input_distance instead.
    :param min_dist: min_dist for UMAP
    :param force_recompute: whether to recompute the embedding even if it exists
    :return: pairwise distance matrix of the UMAP embedding
    """
    # compute kNN graph and distances
    if metric is not None:
        input_distance = metric
    knn_graph = kNN_graph(x, k=k, metric=input_distance).cpu().numpy()
    knn_dists = kNN_dists(x, k=k, metric=input_distance).cpu().numpy()

    # add self node as UMAP requires it
    knn_graph = np.concatenate([np.arange(len(x))[:, None], knn_graph], axis=1)
    knn_dists = np.concatenate([np.zeros(len(x))[:, None], knn_dists], axis=1)

    # comptue PCA for initialization
    pca2 = PCA(n_components=2).fit_transform(x)

    # get umap embedding
    file_name = f"umap_{dataset}_k_{k}_metric_{input_distance}_epochs_{n_epochs}_seed_{seed}_min_dist_{min_dist}_init_pca.pkl"
    recompute = force_recompute
    # try to load the embedding from file, if it exists and recomputation is not required
    if not recompute:
        try:
            with open(os.path.join(root_path, file_name), "rb") as f:
                umapper = pickle.load(f)
            embd = umapper.embedding_
        except FileNotFoundError:
            recompute = True
    if recompute:
        # compute the UMAP embedding with PCA initialization
        umapper = UMAP(
            n_components=2,
            n_neighbors=knn_graph.shape[1],  # note that these are k actual neighbors and the node itself
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_state=seed,
            init=pca2,
            precomputed_knn=(knn_graph, knn_dists)
        )
        embd = umapper.fit_transform(X=x)
        with open(os.path.join(root_path, file_name), "wb") as f:
            pickle.dump(umapper, f, pickle.HIGHEST_PROTOCOL)
    # return distances
    return squareform(pdist(embd))

class keopsKNNIndex(KNNIndex):
    """
    Class for computing the kNN graph with keops in the format required by openTSNE.
    """
    VALID_METRICS = ["euclidean", "cosine", "correlation"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__data = None

    def build(self):
        knn_graph = kNN_graph(self.data, self.k, self.metric).detach().cpu().numpy()
        knn_dists = kNN_dists(self.data, self.k, self.metric).detach().cpu().numpy()
        return knn_graph, knn_dists


def get_tsne_dist(x, perplexity, sim_to_dist="neg_log", input_distance="euclidean"):
    """
    Computes tSNE graph distance
    :param x: data
    :param perplexity: perplexity parameter
    :param sim_to_dist: how to transform the similarities to distances, must be one of "neg_log" or "neg"
    :param input_distance: metric to use for kNN graph, must be one of "euclidean", "cosine", "correlation"
    :return: pairwise distance matrix
    """
    # compute kNN graph using trice the perplexity as the number of neighbors
    k = 3 * perplexity
    knn_index = keopsKNNIndex(x, k, metric=input_distance)

    # compute the affinities
    affinities = PerplexityBasedNN(knn_index=knn_index, perplexity=perplexity).P.tocoo()

    assert affinities.data.min() >= 0, "negative similarities"

    # turn similarities into dissimilarities
    if sim_to_dist == "neg_log":
        affinities.data = -np.log(affinities.data)
        dists = sim_to_dense(affinities)
    elif sim_to_dist == "neg":
        affinities.data = - affinities.data
        dists = sim_to_dense(affinities)
    else:
        raise NotImplementedError("sim_to_dist must be neg_log or neg")

    return dists


def get_tsne_embd_dist(x,
                       perplexity,
                       root_path,
                       dataset,
                       n_epochs=500,
                       n_early_epochs=250,
                       seed=0,
                       rescale_tsne=True,
                       exaggeration=None,
                       input_distance="euclidean",
                       force_recompute=False):
    """
    computes distances in 2D tsne embedding. Also saves the tsne embedding itself, so that dataset and root path are necessary.
    :param perplexity: perplexity for tsne affinities
    :param dataset: name of dataset for file name
    :param root_path: path to project directory, for file name
    :param n_epochs: Number of normal optimization epochs
    :param n_early_epochs: Number of early exaggeration epochs
    :param seed: Random seed
    :param rescale_tsne: Whether to rescale the PCA initialization
    :param exaggeration: exaggeration in tsne
    :param input_distance: metric to use for kNN graph, must be one of "euclidean", "cosine", "correlation"
    :param force_recompute: whether to recompute the tsne embedding even if it exists
    :return: pairwise distances of the tsne embedding
    """
    pca2 = PCA(n_components=2).fit_transform(x)  # note that we use the PCA even for non-euclidean input_distance

    # compute affinities
    k = 3 * perplexity
    knn_index = keopsKNNIndex(x, k, metric=input_distance)
    affinities = PerplexityBasedNN(knn_index=knn_index,
                                   perplexity=perplexity)

    assert affinities.P.tocoo().data.min() >= 0, "negative similarities"

    # optinal rescaling of the PCA initialization
    if rescale_tsne:
        pca_tsne = pca2 / np.std(pca2[:, 0]) / 10000
    else:
        pca_tsne = pca2

    file_name = os.path.join(root_path,
                             f"tsne_{dataset}_perplexity_{perplexity}_n_epochs_{n_epochs}_n_early_epochs_{n_early_epochs}_seed_{seed}"
                             f"_init_pca_rescale_{rescale_tsne}.pkl")
    if exaggeration is not None:
        file_name = file_name.replace(".pkl", f"_exaggeration_{exaggeration}.pkl")

    recompute = force_recompute
    # try to load the embedding unless force_recompute
    if not recompute:
        try:
            tsne_data = load_dict(file_name)
            embd = tsne_data["embds"][-1]
        except FileNotFoundError:
            recompute = True
    if recompute:
        # compute tsne embedding using wrapper that stores some meta data such as embedding during optimization
        tsne = TSNEwrapper(perplexity=None,
                           metric="euclidean",
                           n_jobs=5,  # n_jobs=-10 does not work well, the cell does not print anything
                           random_state=seed,
                           verbose=True,
                           n_iter=n_epochs,
                           early_exaggeration_iter=n_early_epochs,
                           callbacks_every_iters=1,
                           log_kl=False,
                           log_embds=True,
                           log_Z=False,
                           exaggeration=exaggeration
                           )

        embd = tsne.fit_transform(X=x,
                                  affinities=affinities,
                                  initialization=pca_tsne
                                  )
        save_dict(tsne.aux_data, file_name)

    # compute distances in the embedding
    return squareform(pdist(embd))




def get_dist(x=None, distance="euclidean", input_distance="euclidean", **kwargs):
    """
    Wrapper for all distances.
    :param x: data
    :param distance: distance to compute, must be one of "euclidean", "cosine", "correlation", "sknn_hop", "sknn_dist",
    "dtm", "fermat", "core", "eff_res", "deg", "spectral", "diffusion", "umap", "umap_embd", "tsne", "tsne_embd"
    :param input_distance: input distance for all distance other than "euclidean", "cosine", "correlation"
    :param kwargs: key word arguments for the distance function
    :return: distance matrix
    """
    if distance in ["euclidean", "cosine", "correlation"]:
        assert x is not None, "x must be provided for euclidean, cosine and correlation distances"
        dist = squareform(pdist(x, metric=distance))
    elif distance == "sknn_hop":
        dist = get_sp_sknn(x, weighted=False, input_distance=input_distance, **kwargs)
    elif distance == "sknn_dist":
        dist = get_sp_sknn(x, weighted=True, input_distance=input_distance, **kwargs)
    elif distance == "dtm":
        dist = get_dtm_weights(x, input_distance=input_distance, **kwargs)
    elif distance == "fermat":
        dist = get_fermat_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "core":
        dist = get_core_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "eff_res":
        dist = get_eff_res(x, input_distance=input_distance, **kwargs)
    elif distance == "deg":
        dist = get_deg_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "spectral":
        dist = get_spectral_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "diffusion":
        dist = get_diffusion_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "umap":
        dist = get_umap_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "umap_embd":
        dist = get_umap_embd_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "tsne":
        dist = get_tsne_dist(x, input_distance=input_distance, **kwargs)
    elif distance == "tsne_embd":
        dist = get_tsne_embd_dist(x, input_distance=input_distance, **kwargs)
    else:
        raise NotImplementedError
    return dist