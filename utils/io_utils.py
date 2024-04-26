import numpy as np
import os
import copy
import multiprocessing
from functools import partial


def parse_lines(lines, dim):
    # parse line of a ripser result file
    out = {}
    for i, line in enumerate(lines):
        # split bar codes from representatives, remove brackets and handle potentially infinite death times
        bc, reps = line.split(":")
        birth, death = bc.translate({ord(i): None for i in '[]:{}()'}).split(",")
        if death == " ":
            death = np.inf

        # strip brackets and split by commas
        reps = reps.translate({ord(i): None for i in ' [{}()\n'})
        reps = reps.translate({ord(i): ',' for i in ']'}).split(",")

        if dim == 0:
            reps = reps[::2]  # every other entry is empty, so omit it.
            reps = reps if len(reps) == 2 else [reps[0], reps[0]]   # undying component is represented as self loop
            reps = np.stack(reps, axis=0).astype(int)
            out[i] = {"birth": float(birth),
                      "death": float(death),
                      "nodes": reps}
        elif dim == 1:
            reps = np.array(reps).reshape(-1, 3)[:, :2].astype(int)
            out[i] = {"birth": float(birth),
                      "death": float(death),
                      "edges": reps}
        elif dim == 2:
            reps = np.array(reps).reshape(-1, 4)[:, :3].astype(int)
            out[i] = {"birth": float(birth),
                      "death": float(death),
                      "triangles": reps}
        else:
            raise ValueError("dim must be 0, 1 or 2")
    return out


def read_ripser_result(file_name):
    # read a ripser result file and return a dictionary with the bar codes and cycles
    with open(file_name, "r") as file:
        lines = file.readlines()

    # parse lines by dimension starting from the back, since some diagrams do not have dim 2.
    if "persistence intervals in dim 2:\n" in lines:
        pre_2 = lines.index("persistence intervals in dim 2:\n")
        res_2 = parse_lines(lines[pre_2+1:], dim=2)
    else:
        pre_2 = len(lines)
        res_2 = {}

    if "persistence intervals in dim 1:\n" in lines:
        pre_1 = lines.index("persistence intervals in dim 1:\n")
        res_1 = parse_lines(lines[pre_1+1:pre_2], dim=1)
    else:
        pre_1 = pre_2
        res_1 = {}

    if "persistence intervals in dim 0:\n" in lines:
        pre_0 = lines.index("persistence intervals in dim 0:\n")
        # for some reason there always seems to be two lines with "persistence intervals in dim 0:\n", hence shift by 2
        res_0 = parse_lines(lines[pre_0+2: pre_1], dim=0)
    else:
        res_0 = {}

    dgms = [
        np.array([np.array([res_0[k]["birth"], res_0[k]["death"]]) for k in res_0.keys()]),
        np.array([np.array([res_1[k]["birth"], res_1[k]["death"]]) for k in res_1.keys()]),
        np.array([np.array([res_2[k]["birth"], res_2[k]["death"]]) for k in res_2.keys()])
    ]

    cycles = [
        [res_0[k]["nodes"] for k in res_0.keys()],
        [res_1[k]["edges"] for k in res_1.keys()],
        [res_2[k]["triangles"] for k in res_2.keys()]
    ]

    return {"dgms": dgms, "cycles": cycles}


def load_single_res(dataset,
                    n,
                    d,
                    distance,
                    distance_kwargs,
                    sigma,
                    seed,
                    root_path,
                    n_outliers=0,
                    bounding_box=True,
                    rforsigma=None,
                    perturbation=None,
                    repeat=0):
    # load a single experiment
    if isinstance(distance_kwargs, dict):
        dist_str = distance + dist_kwargs_to_str(distance_kwargs)
    else:
        dist_str = distance_kwargs
    n_str = f"{n}_" if n is not None else ""
    d_str = f"d_{d}_" if d is not None else ""
    sigma_str = f"ortho_gauss_sigma_{sigma}_" if sigma is not None else ""
    if n_outliers > 0:
        outlier_str = f"outliers_{n_outliers}_"
        if rforsigma is None:
            if not bounding_box:
                outlier_str = outlier_str + "bounding_box_False_"
        else:
            outlier_str = outlier_str + f"rforsigma_{rforsigma}_"
    else:
        outlier_str = ""

    if perturbation is not None:
        perturbation_str = f"{perturbation}_{repeat}_"
    else:
        perturbation_str = ""

    file_name = os.path.join(root_path,
                             dataset,
                             f"{dataset}_{n_str}{d_str}{sigma_str}{outlier_str}seed_{seed}_{perturbation_str}{dist_str}_rep")

    try:
        res = read_ripser_result(file_name)
        return res
    except FileNotFoundError:
        print("FileNotFountError:")
        print(file_name)
        print("\n")
        res = None
    except ValueError:
        print("ValueError:")
        print(file_name)
        print("\n")
        res = None
    return res


def load_all_sigma_res(sigma,
                       dataset,
                       n,
                       embd_dim,
                       distance,
                       distance_kwargs,
                       seeds,
                       root_path,
                       n_outliers=0,
                       bounding_box=True,
                       rforsigma=None,
                       perturbation=None,
                       repeats=0):
    # loop over all seeds and load the corresponding results
    res_sigma = {}
    for seed in seeds:
        res_seed = {}

        for repeat in range(max(repeats, 1)):
            res = load_single_res(dataset,
                                  n,
                                  embd_dim,
                                  distance,
                                  distance_kwargs,
                                  sigma,
                                  seed,
                                  root_path,
                                  n_outliers,
                                  bounding_box=bounding_box,
                                  rforsigma=rforsigma,
                                  perturbation=perturbation,
                                  repeat=repeat)
            res_seed[repeat] = res

        if repeats > 1:
            res_sigma[seed] = res_seed
        else:
            res_sigma[seed] = res_seed[0]

    if len(seeds) > 1:
        return sigma, res_sigma
    else:
        return sigma, res_sigma[seeds[0]]


def load_multiple_res(datasets,
                      distances,
                      root_path,
                      n=None,
                      embd_dims=None,
                      sigmas=None,
                      seeds=None,
                      n_threads=10,
                      nbs_outliers=0,
                      bounding_box=True,
                      rforsigma=None,
                      perturbation=None,
                      repeats=25):
    """
    Loads multiple experiments. If an argument is a list, it will be iterated over. If it is not a list, it will be used
     for all experiments and the corresponding level in the output dictionaries hierarchy will be flattended.
    :param datasets: dataset name(s) (list or str)
    :param n: number of points (list or int)
    :param embd_dims: embedding dimension (list or int)
    :param distances: dictionary of distance names and distance kwargs (dict)
    :param root_path: root directory of the exerpiments (str)
    :param sigmas: sigma (list, np.ndarray, or float)
    :param seeds: seed (list, np.ndarray, or int)
    :param n_threads: number of threads used to parallelize the loading of different sigma values (int)
    :return: dictionary of results
    """
    # copy distance so that we can modify it in case there is no list of kwargs for a distance
    distances = copy.deepcopy(distances)

    # make arguments lists if they are not iterable over
    if not isinstance(datasets, list):
        datasets = [datasets]

    if not isinstance(embd_dims, list):
        embd_dims = [embd_dims]

    if not (isinstance(sigmas, list) or isinstance(sigmas, np.ndarray)):
        sigmas = [sigmas]

    if not (isinstance(seeds, list) or isinstance(seeds, np.ndarray)):
        seeds = [seeds]

    if not isinstance(nbs_outliers, list) or isinstance(nbs_outliers, np.ndarray):
        nbs_outliers = [nbs_outliers]

    if perturbation is None:
        repeats = 1

    # loop over everything and load results
    all_res = {}
    for dataset in datasets:
        res_dataset = {}
        for embd_dim in embd_dims:
            res_embd_dim = {}
            for distance in distances:
                res_distance = {}
                if not isinstance(distances[distance], list):
                    distances[distance] = [distances[distance]]
                for dist_kwargs in distances[distance]:
                    if isinstance(dist_kwargs, dict):
                        dist_str = distance + dist_kwargs_to_str(dist_kwargs)
                    else:
                        dist_str = dist_kwargs
                    res_dist_kwargs = {}
                    for n_outliers in nbs_outliers:
                        res_n_outliers = {}
                        # if only one thread is used, continue looping over sigmas and seeds in the same thread,
                        # otherwise start a threadpool
                        if n_threads == 1:
                            for sigma in sigmas:
                                sigma, res_sigma = load_all_sigma_res(sigma=sigma,
                                                                      dataset=dataset,
                                                                      n=n,
                                                                      embd_dim=embd_dim,
                                                                      distance=distance,
                                                                      distance_kwargs=dist_kwargs,
                                                                      seeds=seeds,
                                                                      root_path=root_path,
                                                                      n_outliers=n_outliers,
                                                                      bounding_box=bounding_box,
                                                                      rforsigma=rforsigma,
                                                                      perturbation=perturbation,
                                                                      repeats=repeats
                                                                      )
                                res_n_outliers[sigma] = res_sigma
                        else:
                            with multiprocessing.Pool(processes=n_threads) as pool:
                                thread_function = partial(load_all_sigma_res,
                                                          dataset=dataset,
                                                          n=n,
                                                          embd_dim=embd_dim,
                                                          distance=distance,
                                                          distance_kwargs=dist_kwargs,
                                                          seeds=seeds,
                                                          root_path=root_path,
                                                          n_outliers=n_outliers,
                                                          bounding_box=bounding_box,
                                                          rforsigma=rforsigma,
                                                          perturbation=perturbation,
                                                          repeats=repeats)
                                #res_n_outliers = dict(pool.map(thread_function,  sigmas))
                                res_n_outliers = dict(pool.map(thread_function,  sigmas))
                            print(f"Done with {dataset} {embd_dim} {dist_str} n_outliers={n_outliers}, perturbation={perturbation}")

                        # collect results and flatten the hierarchy if there is only one value at a given level
                        if len(sigmas) > 1:
                            res_dist_kwargs[n_outliers] = res_n_outliers
                        else:
                            res_dist_kwargs[n_outliers] = res_n_outliers[0]

                    if len(nbs_outliers) > 1:
                        res_distance[dist_str] = res_dist_kwargs
                    else:
                        res_distance[dist_str] = res_dist_kwargs[nbs_outliers[0]]
                res_embd_dim[distance] = res_distance
            res_dataset[embd_dim] = res_embd_dim
        if len(embd_dims) > 1:
            all_res[dataset] = res_dataset
        else:
            all_res[dataset] = res_dataset[embd_dims[0]]
    if len(datasets) > 1:
        return all_res
    else:
        return all_res[datasets[0]]


def write_lower_tri_dissim(dense_dissim, file_name):
    # write array of distances to file in lower triangular format
    n = dense_dissim.shape[0]
    idx = np.tri(n, n, -1)
    with open(file_name, "w") as file:
        lines = []
        for row_idx, dists_row in zip(idx, dense_dissim):
            lines.append(",".join(dists_row[row_idx.astype(bool)].astype("str")) + "\n")
        file.writelines(lines)


def dist_kwargs_to_str(distance_dict):
    # transform a dictionary of distance kwargs to a string
    return ''.join(['_'+key+'_'+str(distance_dict[key]) for key in distance_dict])


def split_dist_str(dist_str):
    # splits a string and inserts line breaks for better readability
    dist_str = dist_str.split("_")
    if len(dist_str) > 8:
        dist_str.insert(7, "\n")
        dist_str.insert(13, "\n")
    dist_str = "_".join(dist_str)
    return dist_str