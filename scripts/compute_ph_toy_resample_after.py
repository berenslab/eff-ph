import numpy as np
import sys
from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.toydata_utils import get_toy_data
from utils.dist_utils import get_dist
import os
#####################################################################
# compute ph for toy data. First generate n points, then compute the distance matrix,
# then sample 1000 points, then compute the PH.

#see the difference between this and compute_ph_toy_resample_before.py
#####################################################################


dataset = "toy_circle_after"  # must be one of toy_circle, toy_sphere, torus, eyeglasses, inter_circles, toy_blob, two_rings
d = 50  # ambient dimension 2, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000
max_dim = 1  # dimension of highest dimensional topological features computed

sigmas = np.linspace(0.20, 0.60, 81)
sigmas = np.array([np.format_float_positional(sigma, precision=4, unique=True, trim='0') for sigma in sigmas]).astype(float)
seeds = [0,1,2,3,4]
n = 5000

distances = {
    "euclidean": [{}],
    "diffusion": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
    ],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
    ],
}
#####################################################################

root_path = get_path("data")

for seed in seeds:
    for sigma in sigmas:
        # get data
        x = get_toy_data(n=n, dataset=dataset, d=d, seed=0, **{"gaussian": {"sigma": sigma}})
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
                    
                    np.random.seed(seed)
                    sampled = np.random.choice(x.shape[0], 1000, replace=False).tolist()
                    newdists = dists[np.ix_(sampled, sampled)]

                    # compute peristent homology
                    res = compute_ph(newdists, file_name, root_path, dataset, dim=max_dim, delete_dists=True)