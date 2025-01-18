import numpy as np
import sys
import os
import vis_utils
import argparse
from utils.utils import get_path
from utils.io_utils import load_multiple_res, read_ripser_result
from utils.toydata_utils import get_toy_data
from utils.fig_utils import dataset_to_print, plot_dgm_loops, dist_to_print
from utils.pd_utils import sort_cycle
from utils.confidence_utils import median_max_life_time, get_bottleneck_dist, get_bottleneck_dist_matching_test

from vis_utils.loaders import load_dataset
from vis_utils.utils import load_dict, save_dict

root_path = get_path("data")
fig_path = os.path.join(root_path, "figures")
seed = 0
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



def main():
    parser = argparse.ArgumentParser(description="Interactive bottleneck")
    parser.add_argument("--dataset", help="dataset", required=True)
    parser.add_argument("--method", help="method", required=True) #type=str, 
    parser.add_argument("--ratio", help="ratio", required=True)
    parser.add_argument("--seeds", help="seeds", required=False)
    # parser.add_argument("--supdataset", help="supdataset", required=False)

    args = parser.parse_args()

    method = args.method
    seeds = args.seeds
    ratio = args.ratio
    dataset = args.dataset
    # supdataset = args.supdataset #not used anymore since method == resample is not used anymore...
    
    x, y, sknn, pca2, d = load_dataset(root_path, dataset, k=k, seed=0)
    all_res = load_multiple_res(datasets=dataset, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    for i, distance in enumerate(distances):
        for j, full_dist in enumerate(all_res[distance]):
            reference_dgm = all_res[distance][full_dist]
            bottleneck_dists = {}
            checkup_dists = {}
            dgms = []
            if method == "shuffle_pts" or method == "bootstrap":  #not used anymore 
                #then compute the distance between original and 25 shuffled ones
                for r in range(25):
                    file_name = os.path.join(root_path,
                                        dataset,
                                        f"{dataset}_seed_0_{method}_{r}_{full_dist}_rep")
                    dgms.append(read_ripser_result(file_name))
            # elif method == "resample": #not used anymore 
            #     #then compute the distance between seed = 0 and seed = 1+
            #     for r in range(25):
            #         if supdataset == None:
            #             file_name = os.path.join(root_path,
            #                                 dataset,
            #                                 f"{dataset}_seed_{r+1}_{full_dist}_rep")
            #         else:
            #             file_name = os.path.join(root_path,
            #                                 supdataset,
            #                                 f"{supdataset}_seed_{r}_{full_dist}_rep")
            #         dgms.append(read_ripser_result(file_name))
            elif method == "split": #this is the one we wanna use from now on......
                color = dataset.split("_")[3]
                print(color)
                com_datasets = [f"yao_10x_male_{color}_split_{i}" for i in range(10)] + [f"yao_10x_female_{color}_split_{i}" for i in range(5)]
                for com_dataset in com_datasets:
                    file_name = os.path.join(root_path,
                                            com_dataset,
                                            f"{com_dataset}_seed_0_{full_dist}_rep")
                    dgms.append(read_ripser_result(file_name))
            
            
            #############optional, to check ###########
        
            checkup_dists["reference_dgm"] = reference_dgm
            checkup_dists["dgms"] = dgms
            save_dict(checkup_dists, os.path.join(root_path, dataset, f"checkup_dists_{method}_{full_dist}.pkl"))
            #############optional, to check ###########
            if ratio == "quarter":
                the_threshold = int(np.floor(len(reference_dgm["dgms"][1])/4))
            elif ratio == "half":
                the_threshold = int(np.floor(len(reference_dgm["dgms"][1])/2))
            else: #type(ratio) == int:
                print(type(int(ratio)))
                the_threshold = int(np.floor(len(reference_dgm["dgms"][1])/int(ratio)))            

            print(f"Computing bottleneck distances for {full_dist}")
            bottleneck_dists["bottleneck"] = get_bottleneck_dist(dgms, reference_dgm, threshold=the_threshold) 
            
            print(f"Finished {method}, {full_dist}")
            save_dict(bottleneck_dists, os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))
            
            #or if you wanna test if the matching works finely. 
            # bottleneck_dists["bottleneck"], bottleneck_dists["matching"]  = get_bottleneck_dist_matching_test(dgms, reference_dgm, threshold=the_threshold)            
            # save_dict(bottleneck_dists, os.path.join(root_path, dataset, f"bottleneck_match_{method}_{ratio}_{full_dist}.pkl"))
    
if __name__ == "__main__":
    main()