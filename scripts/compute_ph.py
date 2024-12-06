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
d = 50  # ambient dimension 2, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000
max_dim = 1  # dimension of highest dimensional topological features computed

sigmas = np.linspace(0.0, 0.35, 29)
sigmas = np.array([np.format_float_positional(sigma, precision=4, unique=True, trim='0') for sigma in sigmas]).astype(float)

seeds = [0, 1, 2]
n = 1000

distances = {
    "euclidean": [{}],
    "fermat": [
               {"p": 1},
               {"p": 2},
               {"p": 3},
               {"p": 5},
               {"p": 7}
               ],
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
    "core": [
        {"k": 15},
        {"k": 100}
    ],
    "sknn_dist": [
        {"k": 15},
        {"k": 100}
    ],
    "tsne": [
         {"perplexity": 30},
         {"perplexity": 200},
         {"perplexity": 333}
    ],
    "umap": [
         {"k": 100, "use_rho": True, "include_self": True},
         {"k": 999, "use_rho": True, "include_self": True},
    ],
    "tsne_embd": [
        {"perplexity": 8, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True},
        {"perplexity": 30, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True},
        {"perplexity": 333, "n_epochs": 500, "n_early_epochs": 250, "rescale_tsne": True}
    ],
    "umap_embd": [
       {"k": 15, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
       {"k": 100, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
       {"k": 999, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean"},
       {"k": 15, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 3},
       {"k": 100, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 3},
       {"k": 999, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 3},
       {"k": 15, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 5},
       {"k": 100, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 5},
       {"k": 999, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 5},
       {"k": 15, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 10},
       {"k": 100, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 10},
       {"k": 999, "n_epochs": 750, "min_dist": 0.1, "metric": "euclidean", "vis_dim": 10},
    ],
    "eff_res": [
       # normal settings
       {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
       # varying k
       {"corrected": True, "weighted": False, "k": 4, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 30, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 45, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 60, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 75, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 90, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 105, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 120, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 135, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 150, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 300, "disconnect": True},
       {"corrected": True, "weighted": False, "k": 500, "disconnect": True},
    ],
    "diffusion": [
        # normal settings
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
        ## varying t
        {"k": 15, "t": 2, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 4, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 16, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 32, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 128, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 256, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 2, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 4, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 16, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 32, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 128, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 256, "kernel": "sknn", "include_self": False},
        # varying k
        {"k": 4, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 30, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 45, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 60, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 75, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 90, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 105, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 120, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 135, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 150, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 4, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 30, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 45, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 60, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 75, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 90, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 105, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 120, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 135, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 150, "t": 8, "kernel": "sknn", "include_self": False},
    ],
    "potential": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
    ],
    "dpt": [
       {"k": 15, "normalization": "sym", "weighted": False},
       {"k": 100, "normalization": "sym", "weighted": False},
       {"k": 15, "normalization": "symd", "weighted": False},
       {"k": 100, "normalization": "symd", "weighted": False},
       {"k": 15, "normalization": "rw", "weighted": False},
       {"k": 100, "normalization": "rw", "weighted": False}
    ],
    "spectral": [
        {"k": 15, "normalization": "sym", "n_evecs": 2, "weighted": False},
        {"k": 15, "normalization": "sym", "n_evecs": 5, "weighted": False},
        {"k": 15, "normalization": "sym", "n_evecs": 10, "weighted": False},
    ],
    "pca": [
        # normal PCA
        {"n_components": 2, "normalize": False},
        {"n_components": 3, "normalize": False},
        {"n_components": 5, "normalize": False},
        {"n_components": 10, "normalize": False},
        {"n_components": 20, "normalize": False},
        # normalized PCA
        {"n_components": 2, "normalize": True},
        {"n_components": 3, "normalize": True},
        {"n_components": 5, "normalize": True},
        {"n_components": 10, "normalize": True},
        {"n_components": 20, "normalize": True},
    ],
}




#####################################################################

root_path = get_path("data")

for seed in seeds:
    for sigma in sigmas:
        # get data
        x = get_toy_data(n=n, dataset=dataset, d=d, seed=seed, **{"gaussian": {"sigma": sigma}})
        # compute PH
        for distance in distances:
            for dist_kwargs in distances[distance]:
                file_name = f"{dataset}_{n}_d_{d}_ortho_gauss_sigma_{sigma}_seed_{seed}_{distance}" \
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



