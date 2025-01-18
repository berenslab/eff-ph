import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from utils.utils import get_path, read_ripser_result, compute_ph
from utils.io_utils import dist_kwargs_to_str
from utils.dist_utils import get_dist
from vis_utils.loaders import load_dataset
import numpy as np
import argparse


#####################################################################
# THE .PY YOU SHOULD USE TO COMPUTE PH ON ORIGINAL SCRNA DATA
#####################################################################

d = 50  # ambient dimension
max_dim = 1  # dimension of highest dimensional topological features computed

#dataset  options: 

##########1. currently using###############

# "tasic_orange"
# "tasic_purple_lamp5"

# "yao_smart_orange"
# "yao_smart_purple"

# "yao_10x_male_orange_split_[0/1/2/3/4/5/6/7/8/9]"
# "yao_10x_male_purple_split_[0/1/2/3/4/5/6/7/8/9]"

# "yao_10x_female_orange_split_[0/1/2/3/4]"
# "yao_10x_female_purple_split_[0/1/2/3/4]"

##########2. used, but discard from now on###############
######2.1the reason: we decide to take full use of yao's 10x, so subsample is not applied anymore#######

# "yao_10x_male_orange" #full orange.
# "yao_10x_male_orange_20k" #sampled 20k data
# "yao_10x_male_orange_10k"  #sampled 10k data
# "yao_10x_male_orange_0_10k" 
#     #difference between these two: _0_10k i bootstrap from 10k when seed= 0, _10k i resample from the whole 50k from the whole ~50k orange dataset.
# "yao_10x_male_orange_5k"

# "yao_10x_male_purple" #full purple
# "yao_10x_male_purple_20k"
# "yao_10x_male_purple_0_10k"
# "yao_10x_male_purple_5k"

# "yao_10x_female_purple_10k"
# "yao_10x_female_orange_10k"

######2.2the reason: wrong splits. we actually want 10x male to have 5k rather than 10k data, to keep it a similar number to female data.#######
# "yao_10x_male_[orange/purple]_split_10k_[0/1/2/3/4]"

##########3.could be explored, but could also be a dead end.###############

# "yao_10x_male_green_56ITCTX_10k"
# "yao_10x_male_green_56IT_10k"
# "yao_10x_male_green_56IT2_10k"

k = 15

distances = {
    # "euclidean": [{}],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
    ],
    "diffusion": [
        {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
        {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
        {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
    ],
}

#####################################################################

root_path = get_path("data")

def main():
    parser = argparse.ArgumentParser(description="Computing ph on original data")
    parser.add_argument("--dataset", help="dataset", required=True)
    parser.add_argument("--seeds", help="seeds", required=False, default=[0]) 
    # parser.add_argument("--seeds", type=int, nargs='+', help="seeds", required=False, default = 0)

    #no need to sample from 10x here since i should already have the original dataset sampled, and saved....
    
    args = parser.parse_args()

    dataset = args.dataset
    seeds = args.seeds

    # print(seeds)
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
                    res = compute_ph(dists, file_name, root_path, dataset, dim=max_dim, delete_dists=True) #was true......

if __name__ == "__main__":
    main()

