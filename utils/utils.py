import numpy as np
import subprocess
import os
import time
from io_utils import read_ripser_result, write_lower_tri_dissim
from toydata_utils import get_toy_data
from dist_utils import get_dist
from pkg_resources import resource_stream


def get_path(path_type):
    # util for reading in paths from file
    with resource_stream(__name__, "paths") as file:
        lines = file.readlines()

    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: " ".join(line[1:]).strip("\n") for line in lines}

    if path_type == "data":
        try:
            return path_dict["data_path"]
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            return path_dict["fig_path"]
        except KeyError:
            print("There is no path 'fig_path'.")
    elif path_type == "ripser":
        try:
            return path_dict["ripser_path"]
        except KeyError:
            print("There is no path 'ripser_path'.")


def hex_to_rgba(h):
    # transforms color in hexadecimals to rgba
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255. for i in (0, 2, 4)) + (1.0,)


def filter_dtm_dists(outlier_scores):
    # delete all but the best run for dtm for each p_radius
    for p_radius in [1, 2, np.inf]:
        best_full_dist = None
        best_auc = 0
        for full_dist in outlier_scores["dtm"].keys():
            if f"p_radius_{p_radius}" in full_dist:
                auc = outlier_scores["dtm"][full_dist].mean()
                if best_auc < auc:
                    best_full_dist = full_dist
                    best_auc = auc
        keys = list(outlier_scores["dtm"].keys())
        for full_dist in keys:
            if f"p_radius_{p_radius}" in full_dist:
                if full_dist != best_full_dist:
                    del outlier_scores["dtm"][full_dist]
    print(outlier_scores["dtm"].keys())


def compute_ph(dist, file_name, root_dir, dataset, dim=1, delete_dists=True, verbose=True, force_recompute=False, return_time=False):
    """
    Computes the persistent homology of a distance matrix using Ripser. If the risper result already exists, it is just
     read and not recomputed. The distance matrix is written to file first.
    :param dist: distance matrix as numpy array (n, n)
    :param file_name: file_name without extension. Will be used for the distance and the ripser result with different
     extension / suffix
    :param root_dir: root directory for the project
    :param dataset: dataset name
    :param dim: maximal dimension for persistent homology
    :param delete_dists: if True, the distance matrix is deleted after the persistent homology is computed
    :param verbose: if True, some information is printed
    :param force_recompute: if True, persistent homology is recomputed even if the result already exists
    :param return_time: if True, the time for computing the persistent homology is returned
    :return: ripser results dictionary (and run time if return_time)
    """

    recompute = force_recompute
    # if not forced to recompute try to read the result
    if not recompute:
        try:
            res = read_ripser_result(os.path.join(root_dir, dataset, file_name + "_rep"))
        except FileNotFoundError:
            recompute = True
    if recompute:
        assert dist is not None
        # write distance matrix to file
        if verbose:
            print(f"Writing dists for {file_name}")
        file_name_dists = os.path.join(root_dir, dataset, file_name+".lower_distance_matrix")
        write_lower_tri_dissim(dist, file_name_dists)

        # run Ripser
        if verbose:
            print(f"Running Ripser for {file_name}")
        ripser_path = get_path("ripser")

        cmd = f'{ripser_path}/ripser-representatives --dim {dim} {root_dir}/{dataset}/{file_name}.lower_distance_matrix > {root_dir}/{dataset}/{file_name}_rep'

        start_ph = time.time()
        subprocess.run(["bash", "-c", cmd])
        end_ph = time.time()

        # delete distance matrix, which can be large
        if delete_dists:
            if verbose:
                print(f"Deleting dists for {file_name}")
            os.remove(file_name_dists)

        print("\n")
        res = read_ripser_result(os.path.join(root_dir, dataset, file_name+"_rep"))
    if not return_time:
        return res
    else:
        return res, end_ph - start_ph


def measure_run_time(dataset,
                     distance,
                     dist_kwargs,
                     embd_dim,
                     n,
                     seeds,
                     feature_dim,
                     root_path,
                     force_recompute=False,
                     sigma=0.0):
    """
    Utility fucntion for measuring and saving the run time of persistent homology runs.
    :param dataset: dataset name
    :param distance: distance name
    :param dist_kwargs: dictionary of key word arguments for the distance
    :param embd_dim: ambient dimension
    :param n: number of data points
    :param seeds: random seed
    :param feature_dim: highest dimension for which topological features are computed
    :param root_path: root directory of the project
    :param force_recompute: whether a recomputation should be forced, even if the run times for this setting were
    measured before.
    :param sigma: Standard deviation of Gaussian noise in each ambient dimension.
    :return:
    """
    # does not work properly for embedding based method which themselves rely on a seed
    dist_times = []
    ph_times = []
    for seed in seeds:
        base_run_time_file = f"{dataset}_{n}_d_{embd_dim}_feat_dim_{feature_dim}_ortho_gauss_sigma_{sigma}_seed_{seed}_{distance}" \
                        + dist_kwargs_to_str(dist_kwargs) + "_run_time"
        run_time_file = os.path.join(root_path, dataset, base_run_time_file)

        recompute = force_recompute
        if not recompute:
            try:
                with open(run_time_file, "r") as f:
                    lines = f.readlines()
                    dist_times.append(float(lines[0].split(":")[1]))
                    ph_times.append(float(lines[1].split(":")[1]))
            except FileNotFoundError:
                recompute = True
        if recompute:
            data = get_toy_data(n=n, dataset=dataset, d=embd_dim, gaussian={"sigma": sigma}, seed=seed)

            # update the distance with embedding parameters, needed for saving the embedding itself
            dist_kwargs = dist_kwargs.copy()
            if distance.endswith("embd"):
                dist_kwargs.update({"root_path": os.path.join(root_path, dataset),
                                    "dataset": f"n_{n}_d_{embd_dim}_ortho_gauss_sigma_{sigma}",
                                    "seed": seed})
            # get distance
            start_dist = time.time()
            dist = get_dist(data, distance=distance, **dist_kwargs)
            end_dist = time.time()
            t_dist = end_dist - start_dist

            # compute persistent homology and record the run time
            _, t_ph = compute_ph(dist=dist,
                                 file_name="test"+"_"+base_run_time_file,
                                 dim=feature_dim,
                                 delete_dists=True,
                                 force_recompute=True,
                                 root_dir=root_path,
                                 dataset=dataset,
                                 verbose=False,
                                 return_time=True)
            os.remove(os.path.join(root_path, dataset, "test"+"_"+base_run_time_file+"_rep"))

            # write the run times to file
            with open(run_time_file, "w") as f:
                f.write(f"dist: {t_dist}\n")
                f.write(f"ph: {t_ph}\n")
            dist_times.append(t_dist)
            ph_times.append(t_ph)

    return np.array(dist_times), np.array(ph_times)


#####################################################################
# eigenvalue decay functions
#####################################################################


def eff_res_corr_decay(eigenvalues):
    decay = (1-eigenvalues) / np.sqrt(eigenvalues)
    return decay / decay.max()


def eff_res_decay(eigenvalues):
    decay = 1 / np.sqrt(eigenvalues)
    return decay / decay.max()


def lap_eig_decay(eigenvalues, n_evecs=2):
    decay = np.zeros(len(eigenvalues))
    decay[:n_evecs] = 1
    return decay


def diffusion_decay(eigenvalues, t):
    decay = (1-eigenvalues)**t
    return decay / decay.max()
