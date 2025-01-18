import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.dist_utils import get_dist
from vis_utils.loaders import load_dataset
import numpy as np


#####################################################################
# here we test, if we calculate the distance on the whole dataset, then sample, 
# and then compute PH. Accordingly, the sparse_array would better be "True"
# to save memory.  
# see data_yao_10x_male_orange_anosample_sparse.py for the result.
#####################################################################

dataset = "yao_10x_male_orange" 
# or "yao_10x_male_purple". some dataset which includes all the cells but too large to compute PH for all of them

seeds = [0]

distances = {
    "euclidean": [{}],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
    ],
    "diffusion": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False, "sparse_array": True},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False, "sparse_array": True},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False, "sparse_array": True},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False, "sparse_array": True},
    ],
}

max_dim = 1  # maximum homology dimension to compute
k = 15  # needed for sknn graph in load_dataset. Not actually used.
#####################################################################

root_path = get_path("data")


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
                
                np.random.seed(seed)
                sampled = np.random.choice(x.shape[0], 10000, replace=False).tolist()
                newdists = dists[np.ix_(sampled, sampled)]
                # compute PH
                res = compute_ph(newdists, file_name, root_path, dataset, dim=max_dim, delete_dists=True) #was true......



