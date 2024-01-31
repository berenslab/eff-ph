import numpy as np
from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.toydata_utils import get_toy_data
from utils.dist_utils import get_dist
import os
#####################################################################
# Hyperparameters
#####################################################################

dataset = "toy_circle"  # must be one of toy_circle, toy_sphere, torus, eyeglasses, inter_circles, toy_blob, two_rings
d = 50  # ambient dimension
max_dim = 1  # dimension of highest dimensional topological features computed

sigmas = np.linspace(0.0, 0.35, 29)
sigmas = np.array([np.format_float_positional(sigma, precision=4, unique=True, trim='0') for sigma in sigmas]).astype(float)

seeds = [0, 1, 2]

n = 1000

n_outliers = 500


distances = {
    "euclidean": [{}],
    "fermat": [
               #{"p": 1},
               {"p": 2},
               {"p": 3},
               {"p": 5},
               {"p": 7}
               ],
    #"minkowski": [
    #    {"p": 1},
    #    {"p": 2},
    #    {"p": 3},
    #    {"p": 3},
    #    {"p": 5},
    #    {"p": 7},
    #    {"p": np.inf}
    #],
    "dtm": [
            {"k": 4, "p_dtm": 2, "p_radius": 1},
            {"k": 4, "p_dtm": np.inf, "p_radius": 1},
            {"k": 15, "p_dtm": 2, "p_radius": 1},
            {"k": 15, "p_dtm": np.inf, "p_radius": 1},
            {"k": 100, "p_dtm": 2, "p_radius": 1},
            {"k": 100, "p_dtm": np.inf, "p_radius": 1},
            {"k": 4, "p_dtm": 2, "p_radius": 2},
            {"k": 4, "p_dtm": np.inf, "p_radius": 2},
            {"k": 15, "p_dtm": 2, "p_radius": 2},
            {"k": 15, "p_dtm": np.inf, "p_radius": 2},
            {"k": 100, "p_dtm": 2, "p_radius": 2},
            {"k": 100, "p_dtm": np.inf, "p_radius": 2},
            {"k": 4, "p_dtm": 2, "p_radius": np.inf},
            {"k": 4, "p_dtm": np.inf, "p_radius": np.inf},
            {"k": 15, "p_dtm": 2, "p_radius": np.inf},
            {"k": 15, "p_dtm": np.inf, "p_radius": np.inf},
            {"k": 100, "p_dtm": 2, "p_radius": np.inf},
            {"k": 100, "p_dtm": np.inf, "p_radius": np.inf},
    ],
    #"core": [
    #    {"k": 15},
    #    {"k": 100}
    #],
    #"sknn_dist": [
    #    {"k": 15},
    #    {"k": 100}
    #],
    #"tsne": [
    #     {"perplexity": 30},
    #     {"perplexity": 200},
    #     {"perplexity": 333}
    #],
    #"umap": [
    #     {"k": 100, "use_rho": True, "include_self": True},
    #     {"k": 999, "use_rho": True, "include_self": True},
    #],
    #"tsne_embd": [
    #    {"perplexity": 8, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True},
    #    {"perplexity": 30, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True},
    #    {"perplexity": 333, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True}
    #],
    #"umap_embd": [
    #    {"k": 15, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
    #    {"k": 100, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
    #    {"k": 999, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
    #],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True}
    ],
    "diffusion": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
    ],
    #"spectral": [
    #    {"k": 15, "normalization": "none", "n_evecs": 2, "weighted": False},
    #    {"k": 15, "normalization": "none", "n_evecs": 5, "weighted": False},
    #    {"k": 15, "normalization": "none", "n_evecs": 10, "weighted": False},
    #],
}



#####################################################################

root_path = get_path("data")

for seed in seeds:
    # assumes zero-centered data
    x_max_noise = get_toy_data(
        n=n,
        dataset=dataset,
        d=d,
        seed=seed,
        gaussian={"sigma": sigmas.max()},
        )
    r = np.abs(x_max_noise).max()

    for sigma in sigmas:
        # get data
        x = get_toy_data(n=n,
                         dataset=dataset,
                         d=d,
                         seed=seed,
                         gaussian={"sigma": sigma},
                         uniform={"n_noise": n_outliers, "r": r})
        # compute PH
        for distance in distances:
            for dist_kwargs in distances[distance]:
                file_name = f"{dataset}_{n}_d_{d}_ortho_gauss_sigma_{sigma}_outliers_{n_outliers}_rforsigma_{sigmas.max()}_seed_{seed}_{distance}" \
                            + dist_kwargs_to_str(dist_kwargs)
                print(f"Starting with {file_name}")

                # try to load precomputed result
                try:
                    res = read_ripser_result(os.path.join(root_path, dataset, file_name+"_rep"))
                # if non-existent compute PH
                except FileNotFoundError:
                    print(f"Computing PH for {dataset} with sigma {sigma} and distance {distance} with {dist_kwargs}")

                    # copy the dict bc we will change it for the embedding based approaches, so that we can nicely save
                    # the embedding as well
                    dist_kwargs = dist_kwargs.copy()

                    # update the distance with embedding parameters, needed for saving the embedding itself
                    if distance.endswith("embd"):
                        dist_kwargs.update({"root_path": os.path.join(root_path, dataset),
                                            "dataset": f"n_{n}_d_{d}_ortho_gauss_sigma_{sigma}",
                                            "seed": seed})
                    # compute the distance
                    dists = get_dist(x=x, distance=distance, **dist_kwargs)
                    # compute peristent homology
                    res = compute_ph(dists, file_name, root_path, dataset, dim=max_dim, delete_dists=True)



