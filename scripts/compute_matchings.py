from utils.utils import get_path
from utils.io_utils import dist_kwargs_to_str, load_pkl, save_pkl
from utils.toydata_utils import get_toy_data
from utils.dist_utils import get_spectral_diffusion
import os
import time

import sys
module_path = os.path.expanduser(get_path("interval_matching"))
if module_path not in sys.path:
    sys.path.append(module_path)

from match.utils_PH import *
from match.utils_plot import *
from match.utils_data import *

#####################################################################
# Hyperparameters
#####################################################################

dataset = "toy_circle"  # must be one of toy_circle, toy_sphere, torus, eyeglasses, inter_circles, toy_blob, two_rings
d = 50  # ambient dimension 2, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000
max_dim = 1  # dimension of highest dimensional topological features computed

sigmas = [0.1, 0.2, 0.25, 0.3, 0.35] # standard deviation of Gaussian noise
sigmas = np.array([np.format_float_positional(sigma, precision=4, unique=True, trim='0') for sigma in sigmas]).astype(float)

data_seed = 0
seeds = [0, 1, 2]
n = 1000

distances = {
    "euclidean": [{}],

    "diffusion":[
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
    ],
}

ripser_image_path = get_path("ripser_image")
ripser_tight_path = get_path("ripser_tight")

#####################################################################

root_path = get_path("data")
for seed in seeds:
    for sigma in sigmas:

        # compute PH
        for distance in distances:
            for dist_kwargs in distances[distance]:
                x = get_toy_data(n=n, dataset=dataset, d=d, seed=data_seed, **{"gaussian": {"sigma": sigma}})
                y = get_toy_data(n=n, dataset=dataset, d=d, seed=seed + data_seed + 1, **{"gaussian": {"sigma": sigma}})

                if distance == "diffusion":
                    _, x = get_spectral_diffusion(x, return_embd=True, **dist_kwargs)
                    _, y = get_spectral_diffusion(y, return_embd=True, **dist_kwargs)

                # compute persistence and matchings
                file_name = f"{dataset}_{n}_d_{d}_ortho_gauss_sigma_{sigma}_seed_{data_seed}_{seed}_matching_{distance}" \
                            + dist_kwargs_to_str(dist_kwargs)
                print(f"Starting with {file_name}")
                # try to load precomputed result
                try:
                    res = load_pkl(os.path.join(root_path, dataset, file_name))
                # if non-existent compute PH
                except FileNotFoundError:
                    print(f"Computing PH matching for {dataset} with sigma {sigma} and distance {distance} with {dist_kwargs}")

                    # copy the dict bc we will change it for the embedding based approaches, so that we can nicely save
                    # the embedding as well
                    start = time.time()
                    res = matching(x,
                                   y,
                                   dim=1,
                                   affinity_method='A',
                                   check_Morse=True,
                                   file_name=file_name,
                                   root_path=root_path,
                                   ripser_tight_path=ripser_tight_path,
                                   ripser_image_path=ripser_image_path)
                    end = time.time()

                    print(f"Finished in {end-start} seconds")

                    # save the result
                    save_pkl(res, os.path.join(root_path, dataset, file_name))
