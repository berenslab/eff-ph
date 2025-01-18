import numpy as np
import argparse
from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.toydata_utils import get_toy_data
from utils.dist_utils import get_dist
from vis_utils.loaders import load_dataset
from utils.confidence_utils import shuffle_points_per_dim, get_bootstrap
import os
#####################################################################
# this method is to shuffle the distance matrix, which we conclude doesnt make sense at all.
# was first used to make comparison with compute_ph_shuffle_before.py
#####################################################################

d = 50  # ambient dimension
max_dim = 1  # dimension of highest dimensional topological features computed

# seeds = [0, 1, 2]
# seeds = [0]
k = 15


distances = {
    # "euclidean": [{}],
    "diffusion": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
        # {"k": 100, "t": 128, "kernel": "sknn", "include_self": False},
        # {"k": 200, "t": 64, "kernel": "sknn", "include_self": False},
        # {"k": 200, "t": 128, "kernel": "sknn", "include_self": False},
    ],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
        # {"corrected": True, "weighted": False, "k": 200, "disconnect": True},
        # {"corrected": True, "weighted": False, "k": 300, "disconnect": True},
    ],
}
#####################################################################

root_path = get_path("data")


def main():
    parser = argparse.ArgumentParser(description="Interactive bottleneck")
    parser.add_argument("--method", help="method", required=True) #type=str, 
    parser.add_argument("--dataset", help="dataset", required=True)   
    
    args = parser.parse_args()

    method = args.method
    dataset = args.dataset
    
    for seed in seeds:
        x, _, _, _, _ = load_dataset(root_path, dataset, k, seed=0)       
        for r in range(25):
            for distance in distances:
                for dist_kwargs in distances[distance]:
                    file_name = f"{dataset}_seed_{seed}_{method}_{r}_{distance}" + dist_kwargs_to_str(dist_kwargs)
                    print(f"Starting with {file_name}")
                    try:
                        res = read_ripser_result(os.path.join(root_path, dataset, file_name+"_rep"))
                    except FileNotFoundError:

                        # copy the dict bc we will change it for the embedding based approaches, so that we can nicely save
                        # the embedding as well
                        dist_kwargs = dist_kwargs.copy()

                        # update the distance with embedding parameters, needed for saving the embedding itself
                        if distance.endswith("embd"):
                            dist_kwargs.update({"root_path": os.path.join(root_path, dataset),
                                                "dataset": f"",
                                                "seed": seed})

                        print(f"Computing PH for {dataset} seed {seed} with method {method} iteration {r} distance {distance} with {dist_kwargs}")
                        dists = get_dist(x=x, distance=distance, **dist_kwargs)
                        if method == "bootstrap":
                            after_dists = get_bootstrap_matrix_distance(dists, seed=r)
                        elif method == "shuffle_pts":
                            after_dists = shuffle_points_per_dim_matrix(dists, seed=r)
                        else:
                            raise ValueError(f"Unknown perturbation method {method}")
                        res = compute_ph(after_dists, file_name, root_path, dataset, dim=max_dim, delete_dists=True)

if __name__ == "__main__":
    main()