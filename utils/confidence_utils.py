import numpy as np
from utils.pd_utils import filter_diagram, get_life_times
from persim import bottleneck

#########################
# confidence
#########################

def shuffle_dims_per_point(x, seed=0):
    # shuffle dimensions for each point. The set of values for each point stay the same,
    # but they are randomly attributed to dimensions.
    np.random.seed(seed)
    rand = np.random.randn(*x.shape)
    idx = np.argsort(rand, axis=1)
    return np.take_along_axis(x, idx, axis=1)


def rotate_dims_per_point(x, seed=0):
    # roll dimensions. The values for each point are the same,
    # but they are rolled by a random amount, different for each point
    np.random.seed(seed)
    r = np.random.randint(0, x.shape[1], size=x.shape[0])
    idx = np.tile(np.arange(x.shape[1], dtype=int), (x.shape[0], 1))
    idx = (idx + r[:, None]) % x.shape[1]
    return np.take_along_axis(x, idx, axis=1)


def rotate_points_per_dim(x, seed=0):
    # rotate points. The set of values for each dimension stays the same,
    # but the values are attributed to other points in a rolling fashion.
    np.random.seed(seed)
    r = np.random.randint(0, x.shape[0], size=x.shape[1])
    idx = np.tile(np.arange(x.shape[0], dtype=int), (x.shape[1],1)).T
    idx = (idx + r[None]) % x.shape[0]
    return np.take_along_axis(x, idx, axis=0)


def shuffle_points_per_dim(x, seed=0):
    # shuffle points. The set of values for each dimension stays the same,
    # but the values are attributed to other points in a random fashion.
    np.random.seed(seed)
    rand = np.random.randn(*x.shape)
    idx = np.argsort(rand, axis=0)
    return np.take_along_axis(x, idx, axis=0)


def get_bootstrap(x, seed=0):
    n = len(x)
    np.random.seed(seed)
    idx = np.random.choice(n, n, replace=True)
    return x[idx]

####################hzhang added #######################################################

def shuffle_points_per_dim_matrix(dists, seed=0): #done??????
    n = dists.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    upper_values = dists[upper_indices]
    n_upper = len(upper_values)
    
    np.random.seed(seed)
    rand = np.random.randn(n_upper)
    idx = np.argsort(rand, axis=0)
    
    shuffle_upper = np.zeros_like(dists)
    shuffle_upper[upper_indices] = upper_values[idx]
    
    shuffle_matrix =  shuffle_upper + shuffle_upper.T
    return shuffle_matrix

def get_bootstrap_matrix_pts(dists, seed=0):#not sure if its correct...
    n = dists.shape[0]
    np.random.seed(seed)
    sampled = np.random.choice(n, n, replace=True).tolist()
    return dists[np.ix_(sampled, sampled)]

def get_bootstrap_matrix_distance(dists, seed=0):#done.
    n = dists.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    upper_values = dists[upper_indices]
    n_upper = len(upper_values)
    
    np.random.seed(seed)
    idx = np.random.choice(n_upper, n_upper, replace=True).tolist()
    
    boot_upper = np.zeros_like(dists)
    boot_upper[upper_indices] = upper_values[idx]
    
    boot_matrix =  boot_upper + boot_upper.T
    
    return boot_matrix


####################hzhang added #######################################################

def get_bottleneck_dist(dgms, reference_dgm, threshold=0.0):

    # ensure dgms is a list
    if not isinstance(dgms, list):
        dgms = [dgms]

    # Optionally remove features near diagonal for higher performance.
    # Result will be an upper bound on the bottleneck distance
    if isinstance(threshold, int):
        # only include the threshold many most persistent features
        filtered_ref_dgm, max_del_lt = filter_diagram(reference_dgm, dim=1, n=threshold)
        filtered_dgms, max_del_lts = zip(*[filter_diagram(dgm, dim=1, n=threshold)
                                           for dgm in dgms])
        max_del_lt = max(max(max_del_lts), max_del_lt)

        # need this for error bound
        threshold = max_del_lt

    elif isinstance(threshold, float) and threshold > 0.0:
        # remove features with lifetime below threshold
        filtered_ref_dgm = filter_diagram(reference_dgm, dim=1, threshold=threshold)
        filtered_dgms = [filter_diagram(dgm, dim=1, threshold=threshold)
                         for dgm in dgms]
    else:
        filtered_ref_dgm = reference_dgm
        filtered_dgms = dgms

    # compute bottleneck distance for each bootstrap
    bottlenecks = np.array([bottleneck(dgm["dgms"][1], filtered_ref_dgm["dgms"][1]) for dgm in filtered_dgms]) 
    
    mask = bottlenecks < threshold
    print("what is this ...",mask.sum())

    bottlenecks = np.maximum(bottlenecks, threshold)
    
    # upper bound by the threshold used

    # todo warn if threshold is larger than max bottleneck
    # todo retrun with smaller bottleneck if threshold is larger than max bottleneck
    return bottlenecks
    

def get_bottleneck_dist_matching_test(dgms, reference_dgm, threshold=0.0):

    # ensure dgms is a list
    if not isinstance(dgms, list):
        dgms = [dgms]

    # Optionally remove features near diagonal for higher performance.
    # Result will be an upper bound on the bottleneck distance
    if isinstance(threshold, int):
        # only include the threshold many most persistent features
        filtered_ref_dgm, max_del_lt = filter_diagram(reference_dgm, dim=1, n=threshold)
        filtered_dgms, max_del_lts = zip(*[filter_diagram(dgm, dim=1, n=threshold)
                                           for dgm in dgms])
        max_del_lt = max(max(max_del_lts), max_del_lt)

        # need this for error bound
        threshold = max_del_lt

    elif isinstance(threshold, float) and threshold > 0.0:
        # remove features with lifetime below threshold
        filtered_ref_dgm = filter_diagram(reference_dgm, dim=1, threshold=threshold)
        filtered_dgms = [filter_diagram(dgm, dim=1, threshold=threshold)
                         for dgm in dgms]
    else:
        filtered_ref_dgm = reference_dgm
        filtered_dgms = dgms

    bottlenecks = []
    matchidxs = []
    for dgm in filtered_dgms:
        one_bottleneck, matchidx = bottleneck(dgm["dgms"][1], filtered_ref_dgm["dgms"][1], matching = True) 
        bottlenecks.append(one_bottleneck)
        matchidxs.append(matchidx)
    
    mask = bottlenecks < threshold
    print("what is this ...",mask.sum())

    bottlenecks = np.maximum(bottlenecks, threshold)
    #if mask.sum == 0, then matchidx is trustworthy and its identical to bottleneck. im just testing quarter, so i want it to be 0. just a sanity check...
    return bottlenecks, matchidxs



def bottleneck_quantile(dgms, dgm, alpha, threshold=0.0):
    # compute bottleneck distance
    bottlenecks = get_bottleneck_dist(dgms, dgm, threshold=threshold)
    # return quantile
    return np.quantile(bottlenecks, 1 - alpha)


def bottleneck_std(dgms, dgm, threshold=0.0):
    # compute bottleneck distance
    bottlenecks = get_bottleneck_dist(dgms, dgm, threshold=threshold)
    # return quantile
    return np.std(bottlenecks)


def median_max_life_time(dgms):
    max_life_times = []
    for dgm in dgms:
        max_lt = np.max(get_life_times(dgm, dim=1)) if len(dgm['dgms'][1]) > 0 else 0
        max_life_times.append(max_lt)
    return np.median(max_life_times), np.max(max_life_times)


def get_quantile(dist, q=0.1):
    n = dist.shape[0]
    tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.quantile(dist[tri_mask], q)

