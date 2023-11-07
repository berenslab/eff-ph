import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import copy
from utils.toydata_utils import get_torus, invert_torus, get_eyeglasses_order


#####################################################################
# utils for accessing persistence diagrams and representatives
#####################################################################


def get_persistent_feature_id(res, m=1, dim=1, mode="additive"):
    """
    Get id of the m-th most persistent feature in the diagram
    :param res: result dict of ripser
    :param m: get m-th most persistent feature, using 1-indexing
    :param dim: dimension of the feature
    :param mode: "additive" or "multiplicative" for the lifetime
    :return: id of the m-th most persistent feature
    """
    life_times = get_life_times(res, dim=dim, mode=mode)
    cycle_idx_sorted = np.argsort(life_times)[::-1]
    if m > len(cycle_idx_sorted):
        raise ValueError(f"m is larger than the number of features in the diagram, which is {len(cycle_idx_sorted)}")
    return cycle_idx_sorted[m-1]


def get_persistent_cycle(res, m=1, dim=1, mode="additive"):
    """
    Get the m-th most persistent cycle in the diagram
    :param res: result dict of ripser
    :param dim: dimension of the feature
    :param mode: "additive" or "multiplicative" for the lifetime
    :param m: get m-th most persistent cycle, using 1-indexing
    :return: the m-th most persistent cycle
    """
    id = get_persistent_feature_id(res, m=m, dim=dim, mode=mode)
    return res["cycles"][dim][id]


def get_persistent_feature_idx_and_thresh(res, k=1, dim=1, mode="additive"):
    # get several most persistent feature indices together with just before their death time
    life_times = get_life_times(res, dim=dim, mode=mode)
    idx = np.argpartition(life_times, -k)[-k]
    thresh = 0.99 * life_times[idx] + res["dgm"][idx, 0]
    return idx, thresh


def get_life_times(res, dim=1, mode="additive"):
    # get life times. Additive is difference of death and birth time, multiplicative is ratio
    if res["dgms"][dim].shape[0] == 0:
        return np.array([])
    else:
        if mode == "additive":
            return res["dgms"][dim][:, 1] - res["dgms"][dim][:, 0]
        elif mode == "multiplicative":
            return res["dgms"][dim][:, 1] / (res["dgms"][dim][:, 0] + 1e-10)
        else:
            raise ValueError("Mode not recognized")


def max_life_time(lt, mode="additive"):
    # get the maximal life time
    if len(lt) == 0:
        if mode == "additive":
            return 0
        elif mode == "multiplicative":
            return 1
        else:
            raise ValueError("Mode not recognized")
    else:
        return lt.max()


def std_life_times(dgms):
    # compute std of life times for multiple diagrams
    return np.std(np.array([dgm[:, 1] - dgm[:, 0] for dgm in dgms]), axis=0)

def sort_cycle(cycle, start=None):
    # sort a cycle of edges by the order of their nodes
    if start is None:
        start_edge = cycle[0]
    elif isinstance(start, int):
        # start node is given, find edge containing it:
        node_in_edge = np.array([start in edge for edge in cycle]).astype(bool)
        assert node_in_edge.sum() == 2
        start_edge = cycle[node_in_edge][0]
        if start_edge[0] != start:
            start_edge = start_edge[::-1]
    else:
        assert start in cycle or start[::-1] in cycle
        start_edge = start

    id_first_edge = 0  # nessesary if cycle consists of multiple loops
    sorted_cycle = [start_edge]
    unused_edges = cycle.copy()[np.array([set(start_edge) != set(edge) for edge in cycle])]

    for i in range(len(cycle)-1):
        last_edge = sorted_cycle[-1]
        next_node = last_edge[1]
        if next_node != sorted_cycle[id_first_edge][0]:
            # if loop is not yet closed, look for edge linking next_node
            next_edge = unused_edges[np.array([next_node in edge for edge in unused_edges])][0]
            # swap edge if necessary
            if next_edge[0] != next_node:
                next_edge = next_edge[::-1]
        else:
            # previous loop is closed, so start new one with first unused edge
            id_first_edge = len(sorted_cycle)
            next_edge = unused_edges[0]

        # append next edge to sorted cycle and remove from unused edges
        sorted_cycle.append(next_edge)
        unused_edges = unused_edges[np.array([set(next_edge) != set(edge) for edge in unused_edges])]
    return np.stack(sorted_cycle)


def check_cocycle(cocycle, dists, thresh, p=17, verbose=False):
    # check whether a cocycle is present at a certain filtration value already
    n = len(dists)
    condition = True
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < thresh:
                for k in range(j+1, n):
                    if dists[i, k] < thresh and dists[j,k] < thresh:
                        boundary = cocycle[i, j] - cocycle[i, k] + cocycle[j, k]


                        if p is None:
                            boundary_zero = boundary == 0
                        else:
                            boundary_zero = boundary % p == 0

                        if not boundary_zero:
                            condition = False
                            if verbose:
                                print(f"i,j,k = {i}, {j}, {k}")
                                print(f"cocycle values: {cocycle[i, j]}- {cocycle[i, k]}+ {cocycle[j,k]} = {boundary}")

    return condition


def id_to_id_perm(cocycle, idx_perm):
    # translate between normal indices and indices after ripser.py's subsetting
    cocycle = cocycle.copy()
    d = {idx: i for i, idx in enumerate(idx_perm)}
    for i in range(cocycle.shape[0]):
        cocycle[i, 0] = d[cocycle[i, 0]]
        cocycle[i, 1] = d[cocycle[i, 1]]
    return cocycle


def cohom_decoding(cocycle, x, p, thresh, dists=None, idx_perm=None, do_check=True, verbose=False):
    # decode a 1D cocycle into a function to S^1
    if idx_perm is not None:
        x = x[idx_perm]
        cocycle = id_to_id_perm(cocycle, idx_perm)

    n = len(x)

    # build cocycle as sparse matrix
    cocycle_coo = sp.coo_matrix((cocycle[:, 2], (cocycle[:, 0], cocycle[:, 1])), shape=(len(x), len(x)))
    cocycle_coo = cocycle_coo - cocycle_coo.T

    # move entries to interval around zero (essentially [0, ..., p-1] --> [- (p-1)/2, ..., (p-1)/2]
    shift = (p-p % 2)/2

    cocycle_Z = cocycle_coo.copy()
    cocycle_Z.data = (cocycle_coo.data + shift) % p - shift

    if dists is not None:
        assert dists.shape[0] == dists.shape[1]
        assert dists.shape[1] == len(x)
    else:
        dists = np.sqrt(((x[:, None] - x[None])**2).sum(-1))

    # check if lift to Z was successful
    if do_check:
        # todo update check for sparse cocycle_Z
        assert check_cocycle(cocycle_Z, n, dists, thresh, p, verbose)

    # set up linear system for boundary map C^0 --> C^1

    edge_rows = []
    edge_cols = []

    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] <= thresh:
                edge_rows.append(i)
                edge_cols.append(j)
    edges_coo = sp.coo_matrix((np.ones(len(edge_rows)), (edge_rows, edge_cols)), shape=(n, n))
    n_edges = edges_coo.nnz

    # values of cocycle on relevant edges
    # get mask of indices in intersection
    # has all non-zero entries
    b = edges_coo.copy()
    b.data = np.zeros_like(b.data)

    b_tmp = cocycle_Z.multiply(edges_coo).tocoo()
    b_tmp_row_col = np.stack([b_tmp.row, b_tmp.col], axis=1)
    edges_row_col = np.stack([edges_coo.row, edges_coo.col], axis=1)
    mask = (edges_row_col[:, None] == b_tmp_row_col).all(-1).any(1)

    b.data[mask] = b_tmp.data

    # boundary matrix
    d_row = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    d_col = np.concatenate([edges_coo.row, edges_coo.col])
    d_data = np.concatenate([np.ones(len(edges_coo.row)), - np.ones(len(edges_coo.row))])

    d = sp.coo_matrix((d_data, (d_row, d_col)), shape=(n_edges, n))

    f = lsqr(d, b.data)[0]
    return f % 1, np.stack([edges_coo.row, edges_coo.col], axis=1)



def transform_dgms(dgms_cycles, transformation):
    # transforms the birth and death times by certain function. Currently only sqrt is implemented.
    new_dgms_cycles = copy.deepcopy(dgms_cycles)

    # handle an arbitrary level of dictionaries before keys "dgms" and "cycles"
    if "dgms" not in dgms_cycles:
        for key in dgms_cycles:
            new_dgms_cycles[key] = transform_dgms(dgms_cycles[key], transformation)
    else:
        for i, dgm in enumerate(dgms_cycles["dgms"]):
            if transformation == "sqrt":
                new_dgm = np.sqrt(dgm)
            else:
                raise NotImplementedError
            new_dgms_cycles["dgms"][i] = new_dgm

    return new_dgms_cycles


###########################################################################
# Filtering
###########################################################################


def winding_numbers(cohom_gens, loop):
    # Computes the winding numbers of a candidate loop around a generator for a given cohomology class, viewed as a
    # function from the nodes to S^1 = R/Z. Assumes cohom_gens in units of [0, 1). Different dimensions correspond to
    # different generators of cohomology. The loop is a set of edges.
    # The output will be the integer number of times the loop winds around the cohomological generator.

    # sort loop
    sorted_loop = sort_cycle(loop)

    # compute consecutive differences in GT, make sure to use loop around and choose distance in [-0.5, 0.5)
    differences = cohom_gens[sorted_loop[:, 0]] - cohom_gens[sorted_loop[:, 1]]

    differences[differences > 0.5] -= 1
    differences[differences < -0.5] += 1

    # add consecutive differences; they should sum up to integers
    winding_numbers = np.abs(differences.sum(0))
    assert np.allclose(winding_numbers, np.round(winding_numbers)), "winding numbers are not integers"
    return winding_numbers


def get_cohom_gens(dataset, n, seed):
    # get cohomology generators for a given dataset as function that rises from 0 to 1 when going around the loops of
    # the dataset. I.e. use the correspondence H^1 = Hom(X, R/Z) mod homotopy
    if dataset == "torus":
        data = get_torus(n, seed=seed+1, uniform=True)
        cohom_gens = np.array(invert_torus(data, r=1.0, R=2.0)).T / (2*np.pi)
    elif dataset == "toy_circle":
        cohom_gens = np.arange(1, n+1) / n
        cohom_gens = cohom_gens[:, None]
    elif dataset == "two_circles":
        n1 = n//2
        n2 = n - n1
        cohom_gen1 = np.stack([np.arange(1, n1+1) / n1 , np.zeros(n1)], axis=1)
        cohom_gen2 = np.stack([np.zeros(n2), np.arange(1, n2+1) / n2 ], axis=1)
        cohom_gens = np.concatenate([cohom_gen1, cohom_gen2], axis=0)
    elif dataset == "inter_circles":
        cohom_gens = get_cohom_gens("two_circles", n, seed)
    elif dataset == "eyeglasses":
        cohom_gens = get_eyeglasses_order(n) / n
        cohom_gens = cohom_gens[:, None]
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return cohom_gens



def all_cohom_correct(res, dataset, n, seed, verbose=False):
    # If the dataset has n loops this function checks if the n most persistent loops represent generators of the n GT loops

    # todo: Run time depends strongly on the length of the candidate loops. How to speed this up?

    # get cohomology generators
    cohom_gens = get_cohom_gens(dataset, n, seed)

    n_gt_loops = cohom_gens.shape[1]

    if verbose:
        warnings = []

    # get the n_gt_loops most persistent loops
    if n_gt_loops > len(res["dgms"][1]):
        if verbose:
            warnings.append(f"Only {len(res['dmgs'][1])} loops found, but {n_gt_loops} loops are needed")
            return False, warnings
        else:
            return False
    candidate_loops = [get_persistent_cycle(res, m=i+1, dim=1) for i in range(n_gt_loops)]

    # compute winding numbers
    all_good = True
    wind_list = [winding_numbers(cohom_gens, loop) for loop in candidate_loops]


    # check if all loops have winding number one for at least one generator
    # each candidate loop needs to have a winding number of one for at least one generator
    for i, wind_nbs in enumerate(wind_list):
        one_gen = (1 in wind_nbs)
        all_good = all_good and one_gen
        if verbose and not one_gen:
            # uses the order of loops sorted by life time
            warnings.append(f"Loop {i} does not have winding number 1 for any generator")

    # compute the rank of the resulting matrix to check that the loops generate the correct cohomology group
    mat = np.stack(wind_list)
    rank = np.linalg.matrix_rank(mat)
    correct_rank = (rank == n_gt_loops)
    if verbose and not correct_rank:
        warnings.append(f"Rank of winding number matrix is {rank} instead of {n_gt_loops}")

    all_good = all_good and correct_rank
    if verbose:
        return all_good, warnings
    else:
        return all_good


def filter_diagram(res, dim=1, threshold=None, n=None, dob=None, binary=False, lower_dim=False, cohom=None):
    """
    Filter the diagram by various criteria. Only one of threshold, n, dob, lower_dim, cohom can be specified.
    :param res: result dictionary from ripser
    :param dim: dimension of the diagram to filter
    :param threshold: float. If specified omit all features with lifetime smaller than threshold
    :param n: integer. If specified only keep the n longest lived features
    :param dob: float. Filter by the death / birth ratio of the features.
    If binary is True, keep all features, if a single one has death / birth ratio larger than dob.
    If binary is False, keep just those features with death / birth ratio larger than dob.
    :param binary: bool. If True, keep all features with death / birth ratio larger than dob if a single one has it.
    :param lower_dim:  bool. If True, keep all features with longer lifetime than  the second most peristent feature
    of one dimension lower.
    :param cohom: dict. If not None filter the diagram if the most persistent features do not generate the correct
    cohomology group, as measured by the winding numbers of the features. Should contain a dict keys 'dataset', 'seed',
     and 'n' for the dataset name, the random seed for dataset creation and the number of data points.
    :return: filtered result dictionary
    """
    assert (threshold is not None) + (n is not None) + (dob is not None) + lower_dim + (cohom is not None) == 1,\
        "Only one of threshold, n, dob, lower_dim, cohom can be specified"

    dgm = res["dgms"][dim]
    cycles = res["cycles"][dim]

    if threshold is not None:
        # filter by value of persistence
        mask = get_life_times(res, dim, mode="additive") > threshold

    if n is not None:
        # only keep the n longest lived features
        max_del_lt = 0
        sort_idx = np.argsort(get_life_times(res, dim, mode="additive"))[::-1]

        if n < len(dgm):
            # record life time of delete points for error bound
            max_del_lt = dgm[sort_idx[n], 1] - dgm[sort_idx[n], 0]
        mask = sort_idx[:min(n, len(dgm))]

    if dob is not None:
        # only keep features with a death / birth ratio of at least dob
        mask = get_life_times(res, dim, mode="multiplicative") > dob

        # binary filtering means that if a single point was not filtered out, all points are not.
        if binary and mask.sum() > 0:
            mask = np.ones(len(dgm), dtype=bool)

    if lower_dim:
        # only keep features that live for longer than the second most persistent feature in a dimension lower
        # than the current one
        if dim == 0:
            mask = np.ones(len(dgm), dtype=bool)
        else:
            life_times_lower_dim = get_life_times(res, dim-1, mode="additive")
            life_times_lower_dim = life_times_lower_dim[np.isfinite(life_times_lower_dim)]
            sort_idx = np.argsort(life_times_lower_dim)[::-1]
            life_times = get_life_times(res, dim, mode="additive")

            if len(sort_idx) == 0:
                # there is only a single component in the lower dimension (of inf life time),
                # so keep everything in higher dimension
                mask = np.ones_like(life_times, dtype=bool)
            else:
                mask = life_times > life_times_lower_dim[sort_idx[0]]

        if binary and mask.sum() > 0:
            mask = np.ones(len(dgm), dtype=bool)

    if cohom is not None:
        # only keep those diagrams which find the correct loops as most persistent ones
        assert dim == 1, "cohom filtering only implemented for dim=1"
        correct_loops = all_cohom_correct(res, **cohom)
        mask = np.ones(len(dgm), dtype=bool)
        mask *= correct_loops

    out_res = copy.deepcopy(res)

    out_res["dgms"][dim] = dgm[mask]
    out_res["cycles"][dim] = [cycle for i, cycle in enumerate(cycles) if mask[i]]

    if n is not None:
        return out_res, max_del_lt
    else:
        return out_res


def filter_dgms(dgms, inplace=True, **kwargs):
    # wrapper for filtering several ripser results while maintaining the same keys
    # filter diagrams
    if inplace:
        new_dgms = dgms
    else:
        new_dgms = copy.deepcopy(dgms)

    if "dgms" not in new_dgms:
        for key in new_dgms:
            new_dgms[key] = filter_dgms(dgms[key], **kwargs)
    else:
        new_dgms = filter_diagram(
            res=dgms,
            **kwargs)
    return new_dgms


###################################################################################
# Detection score metric
###################################################################################

def outlier_score(dgm, n_features=1, return_mean=False):
    """
    Compute the detection score of a diagram, i.e., measures the relative gap between n_features-th longest lived
    feature and the next longest lived feature.
    :param dgm: persistence diagram
    :param n_features: number of ground truth features
    :param return_mean: whether to return the mean over the relative life times of all desired features or just the
    relative life time of the last desired feature
    :return: detection score
    """

    # handle edge cases
    if n_features == 0:
        return 1 - outlier_score(dgm, n_features=1, return_mean=return_mean)
    elif len(dgm) == n_features:
        return 1
    elif len(dgm) < n_features:
        return 0
    else:
        # compute life times and indices of most persistent features
        life_times = dgm[:, 1] - dgm[:, 0]
        idx_persistent = np.argsort(life_times)[-n_features:]
        idx_next = np.argsort(life_times)[-n_features-1]

        if return_mean:
            # mean over the relative life times of all desired features
            rel_life_times = (life_times[idx_persistent] - life_times[idx_next][None]) / life_times[idx_persistent]
            return np.mean(rel_life_times)
        else:
            # just show the relative life time of the last desired feature
            return (life_times[idx_persistent[0]] - life_times[idx_next]) / life_times[idx_persistent[0]]


def compute_outlier_scores_recursive(dgms, dim=1, n_features=2, **kwargs):
    # wrapper for recursively computing outlier scores. Puts all levels of the hierarchy into a single array, but
    # preserves the hierarchy levels by as dimension order.
    if "dgms" in dgms:
        return outlier_score(dgms["dgms"][dim], n_features=n_features, **kwargs)
    else:
        return [compute_outlier_scores_recursive(dgms[key], dim=dim, n_features=n_features,
                                                 **kwargs) for key in dgms]


def compute_outlier_scores(dgms, dim=1, n_features=1, **kwargs):
    # Wrapper for computing outlier scores for a hierarchy of persistence diagrams. The first two levels of the
    # hierarchy are preserved as dictionary, the remaining ones are combined into a single array.
    outlier_scores = {}
    for key1 in dgms:
        outlier_scores[key1] = {key2: np.array(
            compute_outlier_scores_recursive(dgms[key1][key2], dim=dim,
                                             n_features=n_features, **kwargs)) for key2 in dgms[key1]}
    return outlier_scores


def get_outlier_scores_best_auc(all_res, dim=1, n_features=1):
    # computes the outlier scores and those methods with best area under the curve
    outlier_scores = {}
    auc = {}
    for dist in all_res:
        outlier_scores[dist] = {}
        auc[dist] = {}
        for full_dist in all_res[dist]:
            outlier_scores[dist][full_dist] = compute_outlier_scores_recursive(dgms=all_res[dist][full_dist],
                                                                               dim=dim,
                                                                               n_features=n_features)
            auc[dist][full_dist] = outlier_scores[dist][full_dist].mean()
    best_aucs = {}
    for dist in all_res:
        for full_dist in auc[dist]:
            if dist not in best_aucs:
                best_aucs[dist] = {"run": full_dist, "auc": auc[dist][full_dist]}
            else:
                if auc[dist][full_dist] > best_aucs[dist]["auc"]:
                    best_aucs[dist] = {"run": full_dist, "auc": auc[dist][full_dist]}
    return outlier_scores, best_aucs



