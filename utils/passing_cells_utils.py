import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils.utils import get_path
from utils.pd_utils import sort_cycle, get_life_times
from vis_utils.utils import load_dict, save_dict


def compute_density(cell_list, cluster_template):
    lens = len(cell_list)
    density = []
    for cluster in cluster_template:
        count = cell_list.count(cluster)
        density.append(count)
    density = np.array(density)/lens
    return density


def reorder_yaxis(item):
    number, letter = item.split('_')
    return (letter, 300-int(number))


def fig_distribution(all_res, cell_group_name, clusters, full_d, mask, save_fig = False):
    # test with 7 most persistent loops for each distance
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])
    n_cols = 7
    mid_col = np.floor(n_cols/2)

    color_list = ['#d714fb', '#006100', '#5d0014', '#75ff3d', '#ffcef3', '#656979', '#db0c59']

    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, constrained_layout=True, figsize=(10, 10))
    row = 0

    passing_cell_dict = {f"{clusters}": clusters}
    
    if "10x" not in cell_group_name:
        masked_type = full_d['clusters'][mask]
    else:
        np.random.seed(0)
        mm = np.random.choice(len(full_d['clusters'][mask]), 20000, replace=False)
        masked_type = full_d['clusters'][mask][mm]

    for i, distance in enumerate(all_res.keys()):
        for j, full_dist in enumerate(all_res[distance]):
            passing_cell_dict[full_dist] = []
            
            res = all_res[distance][full_dist]   
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1][:n_cols]
            
            for k in range(n_cols):
                edge = res["cycles"][1][loop_idx_sorted[k]]
                cell_type_idx = list(set(edge.flatten()))
                
                loop_type_idx = masked_type[cell_type_idx]
                loop_type_idx = sorted(loop_type_idx)
                
                cell_type = list(full_d['clusterNames'][loop_type_idx])
                
                if "tasic" in cell_group_name:
                    try:
                        tasic_to_yao = load_dict(os.path.join(get_path("data") , "tasic_to_yao.pkl"))
                    except FileNotFoundError:
                        print("could not find tasic_to_yao.pkl")
                    cell_type = [tasic_to_yao[tasic_type] for tasic_type in cell_type]

                density = compute_density(cell_type, clusters)
                passing_cell_dict[full_dist].append(density)
                
                weights= np.ones_like(loop_type_idx) / len(loop_type_idx)
                
                ax[row][k].hist(cell_type, bins = len(set(loop_type_idx)), weights = weights, orientation='horizontal', color = color_list[k]) #color_list[k]
                ax[row][k].set_box_aspect(aspect=3)
            ax[row][3].set_title(full_dist, fontsize=5)   
            row+=1
    fig.suptitle(f"Distribution, {cell_group_name}", fontsize = 16)
    # fig.autofmt_xdate()
    # plt.tight_layout()    
    plt.show()
    if save_fig == True:
        fig.savefig(os.path.join(get_path("figures"), f"distribution_{cell_group_name}.png"), dpi=300)
        
        return passing_cell_dict
    
    
def fig_loops_curve(all_res, n_loops, cell_group_name, clusters, full_d, mask, smooth = True, save_fig = False):
    
    # test with 7 most persistent loops for each distance
    
    
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])
    plot_loops = True

    color_list = ['#d714fb', '#006100', '#5d0014', '#75ff3d', '#ffcef3', '#656979', '#db0c59']
    
    fig, ax = plt.subplots(nrows=n_exp, ncols = 1, constrained_layout=True, figsize=(10, 21))
    row = 0
    masked_type = full_d['clusters'][mask]

    for i, distance in enumerate(all_res.keys()):
        for j, full_dist in enumerate(all_res[distance]):        
            res = all_res[distance][full_dist] 

            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1][:n_loops]
            for k in range(n_loops):
                edge = res["cycles"][1][loop_idx_sorted[k]]
                sorted_cycle = sort_cycle(edge)
                
                loop_type_idx = masked_type[sorted_cycle]
                
                cell_type = list(full_d['clusterNames'][loop_type_idx][:,0])
                
                loop_type_idx = masked_type[sorted_cycle]
            
                if "tasic" in cell_group_name:
                    try:
                        tasic_to_yao = load_dict(os.path.join(get_path("data") , "tasic_to_yao.pkl"))
                        cell_type = [tasic_to_yao[tasic_type] for tasic_type in cell_type]
                    except FileNotFoundError:
                        print("could not find tasic_to_yao.pkl")
                                  
                cell_type_mapped = [clusters.index(value) for value in cell_type]
                min_index = cell_type_mapped.index(min(cell_type_mapped))
                cell_type_mapped_reorder = cell_type_mapped[min_index:] + cell_type_mapped[:min_index]
                lens = len(cell_type_mapped_reorder)
                if smooth == True:
                    cell_type_mapped_reorder, lens = smooth_curve(cell_type_mapped_reorder)
                    
                x = np.linspace(0,1,lens)
                ax[row].plot(x, cell_type_mapped_reorder, color = color_list[k])

            ax[row].set_yticks(range(len(clusters)))
            ax[row].set_yticklabels(clusters)
                
            ax[row].set_axisbelow(True)
            ax[row].yaxis.grid(color="0.9", linestyle='dashed')
            ax[row].set_title(full_dist, fontsize=5)   
            row+=1
    fig.suptitle(f"{cell_group_name}, first {n_loops} loops", fontsize = 16)
    
    plt.show()
    
    if save_fig == True:
        fig.savefig(os.path.join(get_path("figures"), f"{cell_group_name}_curves_{n_loops}.png"), dpi=300)

def smooth_curve(nums): # only if isolated...
    # if len(nums) < 3:
    #     return nums
    
    result = [nums[0]]  

    for i in range(1, len(nums) - 1):
        if nums[i] == nums[i - 1] or nums[i] == nums[i + 1]:
            result.append(nums[i])

    result.append(nums[-1])
    
    lens = len(result)

    return result, lens


def distribution_max(all_res, cell_group_name, clusters, full_d, mask, geq_maxs):
    row = 0
    passing_cell_dict = {"clusters": clusters, "idx": [], "density": []}
    if "10x" not in cell_group_name:
        masked_type = full_d['clusters'][mask]
    else:
        np.random.seed(0)
        mm = np.random.choice(len(full_d['clusters'][mask]), 10000, replace=False)
        masked_type = full_d['clusters'][mask][mm]
    
    for i, distance in enumerate(all_res.keys()):
        for j, full_dist in enumerate(all_res[distance]):            
            res = all_res[distance][full_dist]   
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1]
            
            for k in range(geq_maxs[row]):
                passing_cell_dict["idx"].append(f"{row}{k}")
                
                edge = res["cycles"][1][loop_idx_sorted[k]]
                cell_type_idx = list(set(edge.flatten()))
                
                loop_type_idx = masked_type[cell_type_idx]
                loop_type_idx = sorted(loop_type_idx)
                
                cell_type = list(full_d['clusterNames'][loop_type_idx])
                
                if "tasic" in cell_group_name:
                    try:
                        tasic_to_yao = load_dict(os.path.join(get_path("data") , "tasic_to_yao.pkl"))
                    except FileNotFoundError:
                        print("could not find tasic_to_yao.pkl")
                    cell_type = [tasic_to_yao[tasic_type] for tasic_type in cell_type]

                density = compute_density(cell_type, clusters)
                passing_cell_dict["density"].append(density)
            row+=1
    passing_cell_dict["density"] = np.vstack( passing_cell_dict["density"])

    return passing_cell_dict



def distribution_split(all_res, cell_group_name, clusters, full_d, mask, geq_maxs, total_num):
    row = 0
    passing_cell_dict = {"clusters": clusters, "idx": [], "density": []}
    
    split_i = int(cell_group_name[-1])
    shuffled_idx = np.arange(sum(mask))
   
    np.random.seed(0)
    np.random.shuffle(shuffled_idx)
    
    shuffled_idx = np.array_split(shuffled_idx, total_num)
    
    masked_type = full_d['clusters'][mask][shuffled_idx[split_i]]
    
    for i, distance in enumerate(all_res.keys()):
        for j, full_dist in enumerate(all_res[distance]):            
            res = all_res[distance][full_dist]   
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1]
            
            for k in range(geq_maxs[row]):
                passing_cell_dict["idx"].append(f"{split_i}{row}{k}")
                
                edge = res["cycles"][1][loop_idx_sorted[k]]
                cell_type_idx = list(set(edge.flatten()))
                
                loop_type_idx = masked_type[cell_type_idx]
                loop_type_idx = sorted(loop_type_idx)
                
                cell_type = list(full_d['clusterNames'][loop_type_idx])

                density = compute_density(cell_type, clusters)
                passing_cell_dict["density"].append(density)
            row+=1
    passing_cell_dict["density"] = np.vstack( passing_cell_dict["density"])

    return passing_cell_dict