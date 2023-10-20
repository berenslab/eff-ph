from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.dist_utils import get_dist
import os
from vis_utils.loaders import load_dataset
import numpy as np


#####################################################################
# Hyperparameters
#####################################################################

# dataset to use, must be one of 'mca_ss2', 'pallium_scVI_IPC_small', 'neurosphere_gopca_small',
# 'hippocampus_gopca_small', 'HeLa2_gopca', or 'pancreas_gopca'
dataset = "mca_ss2"

use_correlation = False  # whether to use correlation distance as input distance (True only for malaria dataset)


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
    ],
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
    "spectral": [
        {"k": 15, "normalization": "none", "n_evecs": 2, "weighted": False},
        {"k": 15, "normalization": "none", "n_evecs": 5, "weighted": False},
        {"k": 15, "normalization": "none", "n_evecs": 10, "weighted": False},
    ],
}


seeds = [0, 1, 2]


max_dim = 1  # maximum homology dimension to compute
k = 15  # needed for sknn graph in load_dataset. Not actually used.
#####################################################################


# add input distance to all non-vanilla distances
if use_correlation:
    for distance in distances:
        if distance in ["euclidean", "cosine", "correlation"]:
            continue
        for dist_kwargs in distances[distance]:
            dist_kwargs["input_distance"] = "correlation"


root_path = get_path("../data")


for seed in seeds:
    # load data
    x, _, _, _, _ = load_dataset(root_path, dataset, k, seed=seed)
    # compute PHs
    for distance in distances:
        for dist_kwargs in distances[distance]:
            file_name = f"{dataset}_seed_{seed}_{distance}" + dist_kwargs_to_str(dist_kwargs)
            print(f"Starting with {file_name}")
            # try to load precomputed result
            try:
                res = read_ripser_result(os.path.join(root_path, dataset, file_name+"_rep"))
            # if non-existent compute PH
            except FileNotFoundError:
                print(f"Computing PH for {dataset} seed {seed} with distance {distance} with {dist_kwargs}")

                # copy the dict bc we will change it for the embedding based approaches, so that we can nicely save
                # the embedding as well
                dist_kwargs = dist_kwargs.copy()

                # update the distance with embedding parameters, needed for saving the embedding itself
                if distance.endswith("embd"):
                    dist_kwargs.update({"root_path": os.path.join(root_path, dataset),
                                        "dataset": f"",
                                        "seed": seed})
                # compute the distance
                dists = get_dist(x=x, distance=distance, **dist_kwargs)
                # compute PH
                res = compute_ph(dists, file_name, root_path, dataset, dim=max_dim, delete_dists=True)



