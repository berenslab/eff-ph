
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils.utils import get_path

from utils.io_utils import read_ripser_result
from utils.confidence_utils import median_max_life_time, get_bottleneck_dist
from utils.fig_utils import dataset_to_print, plot_dgm_loops, dist_to_print, plot_edges_on_scatter
from utils.pd_utils import get_life_times
import matplotlib
import glasbey

from vis_utils.utils import load_dict, save_dict

def fig_most_persistent(all_res, dataset, embd_n_mask, method, save=False):
    style_file = "utils.style"
    plt.style.use(style_file)
    
    root_path = get_path("data")
    fig_path = get_path("figures")
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])
    n_cols = 8
    mid_col = np.floor(n_cols/2)
    plot_loops = True

    width_ratios = [0.75] + [1] * (n_cols-1)

    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, figsize=(n_exp, n_cols), constrained_layout=True, width_ratios=width_ratios)
    seed = 0
    row = 0
    for i, distance in enumerate(all_res.keys()):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]    
            
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            
            total_loop = len(life_array)
              
            dgms = []
            for r in range(25):
                file_name = os.path.join(root_path,
                                    dataset,
                                    f"{dataset}_seed_{seed}_{method}_{r}_{full_dist}_rep")
                dgms.append(read_ripser_result(file_name))
            life_median, life_max = median_max_life_time(dgms)

            if len(res["dgms"][1]) != 0:
                ax[row][int(mid_col)].set_title(full_dist, fontsize=5)
                plot_dgm_loops(res, embd_n_mask, y="k", n_loops=0, confidence = [life_max, life_median], ax=ax[row], plot_only=[1], style=style_file, linewidth=1, plot_loops=plot_loops)
       
            geq_median = np.sum(np.array(reference_life_long) >= life_median)
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            
            ax[row][0].set_title(f">= median {geq_median}, >= max {geq_max}\n total# of loops = {total_loop}", fontsize = 5)
            ax[row][1].set_title("median="+"{:3f}".format(life_median)+ "\n max=" + "{:3f}".format(life_max), fontsize = 5)
            
            ax[row][0].legend().set_visible(False)
            
            for n in range(1, n_cols):
                if dataset == "yao_smart_orange":
                    ax[row][n].set_xlim(-20, 40)
                    ax[row][n].set_ylim(5, 85)
                elif dataset == "yao_smart_purple":
                    ax[row][n].set_xlim(20, 80)
                    ax[row][n].set_ylim(-70, 40)
                elif "10x" in dataset and "orange" in dataset:
                    ax[row][n].set_xlim(-65, -15)
                    ax[row][n].set_ylim(25, 70)
                elif "10x" in dataset and "purple" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-30, 30)
            row+=1
    fig.suptitle(f"{dataset}'s confidence, most persistent loops, {method}, 25 repeats", fontsize = 12)       
    plt.show()
    if save is True:
        fig.savefig(os.path.join(fig_path, f"Most_persistent_{dataset}_{method}.png"), dpi=300)


def fig_bottleneck(all_res, dataset, embd_n_mask, method, ratio = "quarter", save=None):
    '''
    plot the persistant paragrams and 7 loops with the longest life time* # of distances/conditions
    '''
    
    style_file = "utils.style"
    plt.style.use(style_file)
    
    root_path = get_path("data")
    fig_path = get_path("figures")
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])
    n_cols = 8
    mid_col = np.floor(n_cols/2)
    plot_loops = True

    width_ratios = [0.75] + [1] * (n_cols-1)

    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, figsize=(8, 8), constrained_layout=True, width_ratios=width_ratios)
    seed = 0
    row = 0
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            total_loop = len(reference_life_long)
            
            tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))

            bottleneck_dists = list(tryit.values())[0]
            life_median = np.median(bottleneck_dists)
            life_max = np.max(bottleneck_dists)
            
            geq_median = np.sum(np.array(reference_life_long) >= life_median)
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            
            ax[row][0].set_title(f">= median {geq_median}, >= max {geq_max}\n total# of loops = {total_loop}", fontsize = 5)
            ax[row][1].set_title("median="+"{:3f}".format(life_median)+ "\n max=" + "{:3f}".format(life_max), fontsize = 5)

            ax[row][int(mid_col)].set_title(full_dist, fontsize=5)
            plot_dgm_loops(res, embd_n_mask, y="k", n_loops=0, confidence = [life_max, life_median], ax=ax[row], plot_only=[1], style=style_file, linewidth=1, plot_loops=plot_loops)

    
            ax[row][0].legend().set_visible(False)
            
            for n in range(1, n_cols):
                if dataset == "yao_smart_orange":
                    ax[row][n].set_xlim(-20, 40)
                    ax[row][n].set_ylim(5, 85)
                elif dataset == "yao_smart_purple":
                    ax[row][n].set_xlim(20, 80)
                    ax[row][n].set_ylim(-70, 40)
                elif "10x" in dataset and "purple" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-30, 30)
                elif "10x" in dataset and "orange" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-10, 50)
                elif "10x" in dataset and "purple" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-40, 10)
                    ax[row][n].set_ylim(-90, -20)
            
            
            # ax[row][0].set_xticks([])
            # ax[row][0].set_yticks([])
            # ax[row][0].set_ylabel("Death")

            row+=1
    fig.suptitle(f"{dataset}'s confidence, bottleneck, {method}, 25 repeats, {ratio}", fontsize = 12)       

    plt.show()
    if save is True:
        fig.savefig(os.path.join(fig_path, f"Bottleneck_{dataset}_{method}.png"), dpi=300)
    elif type(save) == str:
        fig.savefig(os.path.join(fig_path, f"{save}.png"), dpi=300)
        
        
def fig_bottleneck_max(all_res, dataset, embd_n_mask, method, ratio = "quarter", save=None):
    '''
    plot the persistant paragrams and m loops with the longest life time* # of distances/conditions, 
    wehre m is the number of loops/features with life time >= max bottleneck distance.
    '''
    
    style_file = "utils.style"
    plt.style.use(style_file)
    
    root_path = get_path("data")
    fig_path = get_path("figures")
    
    life_maxs, geq_maxs = get_life_geq_max(all_res, dataset, method, ratio)
    
    n_cols = max(geq_maxs)+1
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])

    width_ratios = [0.75] + [1] * (n_cols-1)
    
    tab10 = matplotlib.colormaps.get_cmap("tab10")
    existing_colors = [tab10(1)]
    colors = glasbey.extend_palette(existing_colors, n_cols+len(existing_colors)+2)[len(existing_colors)+1:]
    
    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, figsize=(n_cols, n_exp), constrained_layout=True, width_ratios=width_ratios)
    seed = 0
    row = 0
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            
            ax[row][0].set_title(f">= max {geq_maxs[row]}", fontsize = 5)
            plot_dgm_loops(res, embd_n_mask, y="k", n_loops=0, confidence = life_maxs[row], ax=ax[row], plot_only=[1], style=style_file, linewidth=1, plot_loops=False)
            
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1]
            
            for k in range(1, n_cols):
                ax[row][k].set_title(f"{row}{k-1}", fontsize=5)
                loop_id = loop_idx_sorted[k-1]
                plot_edges_on_scatter(ax=ax[row][k],
                                 edge_idx=res["cycles"][1][loop_id],
                                 x=embd_n_mask,
                                 color=colors[k],
                                  linewidth=1)
                if k < geq_maxs[row]+1:
                    ax[row][0].scatter(*res["dgms"][1][loop_id].T, c=colors[k], s=10)
                else:
                    # ax[row][k].axis("off")
                    ax[row][k].remove()
            
            
            if geq_maxs[row]>0:
                ax[row][1].set_title(f"{full_dist}  \n {row}0 ", fontsize=5)
            ax[row][0].legend().set_visible(False)
            
            for n in range(1, n_cols):
                if dataset == "yao_smart_orange":
                    ax[row][n].set_xlim(-20, 40)
                    ax[row][n].set_ylim(5, 85)
                elif dataset == "yao_smart_purple":
                    ax[row][n].set_xlim(20, 80)
                    ax[row][n].set_ylim(-70, 40)
                elif "10x" in dataset and "orange" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-65, -15)
                    ax[row][n].set_ylim(25, 70)
                elif "10x" in dataset and "purple" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-30, 30)
                elif "10x" in dataset and "orange" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-10, 50)
                elif "10x" in dataset and "purple" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-40, 10)
                    ax[row][n].set_ylim(-90, -20)
            
            ax[row][0].set_xticks([])
            ax[row][0].set_yticks([])
            # ax[row][0].set_ylabel("Death")

            row+=1
    fig.suptitle(f"{dataset}, bottleneck, {method}, max", fontsize = 12)       

    plt.show()
    if save is True:
        fig.savefig(os.path.join(fig_path, f"Bottleneck_max_{dataset}_{method}.png"), dpi=300)
    elif type(save) == str:
        fig.savefig(os.path.join(fig_path, f"{save}.png"), dpi=300)
    
    return geq_maxs


def get_life_geq_max(all_res, dataset, method, ratio = "quarter"):
    root_path = get_path("data")
    geq_maxs = []
    life_maxs = []
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))

            # tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_match_{method}_{ratio}_{full_dist}.pkl"))
            bottleneck_dists = list(tryit.values())[0]
            life_max = np.max(bottleneck_dists)
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            life_maxs.append(life_max)
            geq_maxs.append(geq_max)
 
    return life_maxs, geq_maxs


def fig_bottleneck_median(all_res, dataset, embd_n_mask, method, ratio = "quarter", scale = 2, save=None):
    '''
    plot the persistant paragrams and m loops with the longest life time* # of distances/conditions, 
    wehre m is the number of loops/features with life time >= max bottleneck distance.
    '''
    
    style_file = "utils.style"
    plt.style.use(style_file)
    
    root_path = get_path("data")
    fig_path = get_path("figures")
    
    life_maxs, geq_maxs = get_life_geq_median(all_res, dataset, method, ratio, scale)
    
    n_cols = max(geq_maxs)+1
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])

    width_ratios = [0.75] + [1] * (n_cols-1)
    
    tab10 = matplotlib.colormaps.get_cmap("tab10")
    existing_colors = [tab10(1)]
    colors = glasbey.extend_palette(existing_colors, n_cols+len(existing_colors)+2)[len(existing_colors)+1:]
    
    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, figsize=(n_cols, n_exp), constrained_layout=True, width_ratios=width_ratios)
    seed = 0
    row = 0
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            
            ax[row][0].set_title(f">= max {geq_maxs[row]}", fontsize = 5)
            plot_dgm_loops(res, embd_n_mask, y="k", n_loops=0, confidence = life_maxs[row], ax=ax[row], plot_only=[1], style=style_file, linewidth=1, plot_loops=False)
            
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1]
            
            for k in range(1, n_cols):
                ax[row][k].set_title(f"{row}{k-1}", fontsize=5)
                loop_id = loop_idx_sorted[k-1]
                plot_edges_on_scatter(ax=ax[row][k],
                                 edge_idx=res["cycles"][1][loop_id],
                                 x=embd_n_mask,
                                 color=colors[k],
                                  linewidth=1)
                if k < geq_maxs[row]+1:
                    ax[row][0].scatter(*res["dgms"][1][loop_id].T, c=colors[k], s=10)
                else:
                    # ax[row][k].axis("off")
                    ax[row][k].remove()
            
            
            if geq_maxs[row]>0:
                ax[row][1].set_title(f"{full_dist}  \n {row}0 ", fontsize=5)
            ax[row][0].legend().set_visible(False)
            
            for n in range(1, n_cols):
                if dataset == "yao_smart_orange":
                    ax[row][n].set_xlim(-20, 40)
                    ax[row][n].set_ylim(5, 85)
                elif dataset == "yao_smart_purple":
                    ax[row][n].set_xlim(20, 80)
                    ax[row][n].set_ylim(-70, 40)
                elif "10x" in dataset and "orange" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-65, -15)
                    ax[row][n].set_ylim(25, 70)
                elif "10x" in dataset and "purple" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-30, 30)
                elif "10x" in dataset and "orange" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-10, 50)
                elif "10x" in dataset and "purple" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-40, 10)
                    ax[row][n].set_ylim(-90, -20)
            
            
            ax[row][0].set_xticks([])
            ax[row][0].set_yticks([])
            # ax[row][0].set_ylabel("Death")

            row+=1
    fig.suptitle(f"{dataset}, bottleneck, {method}, {scale}*median", fontsize = 12)       

    plt.show()
    if save is True:
        fig.savefig(os.path.join(fig_path, f"Bottleneck_median2_{dataset}_{method}.png"), dpi=300)
    elif type(save) == str:
        fig.savefig(os.path.join(fig_path, f"{save}.png"), dpi=300)
    
    return geq_maxs


def get_life_geq_median(all_res, dataset, method, ratio = "quarter", scale = 2):
    root_path = get_path("data")
    geq_maxs = []
    life_maxs = []
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))
            if method != "split":
                bottleneck_dists = list(tryit.values())[0]
            elif method == "split":
                dig = int(dataset[-1])
                if "female" in dataset:
                    dig+=10
                # print("dig = ", dig)
                bottleneck_dists = np.hstack((list(tryit.values())[0][:dig], list(tryit.values())[0][dig+1:]))
            life_max = np.median(bottleneck_dists)*int(scale)
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            life_maxs.append(life_max)
            geq_maxs.append(geq_max)
 
    return life_maxs, geq_maxs

def fig_bottleneck_std(all_res, dataset, embd_n_mask, method, ratio = "quarter", scale = 2, save=None):
    '''
    plot the persistant paragrams and m loops with the longest life time* # of distances/conditions, 
    wehre m is the number of loops/features with life time >= max bottleneck distance.
    '''
    
    style_file = "utils.style"
    plt.style.use(style_file)
    
    root_path = get_path("data")
    fig_path = get_path("figures")
    
    life_maxs, geq_maxs = get_life_geq_std(all_res, dataset, method, ratio, scale)
    
    n_cols = max(geq_maxs)+1
    n_exp = sum([len(all_res[distance]) for distance in all_res.keys()])

    width_ratios = [0.75] + [1] * (n_cols-1)
    
    tab10 = matplotlib.colormaps.get_cmap("tab10")
    existing_colors = [tab10(1)]
    colors = glasbey.extend_palette(existing_colors, n_cols+len(existing_colors)+2)[len(existing_colors)+1:]
    
    fig, ax = plt.subplots(nrows=n_exp, ncols=n_cols, figsize=(n_cols, n_exp), constrained_layout=True, width_ratios=width_ratios)
    seed = 0
    row = 0
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            
            ax[row][0].set_title(f">= max {geq_maxs[row]}", fontsize = 5)
            plot_dgm_loops(res, embd_n_mask, y="k", n_loops=0, confidence = life_maxs[row], ax=ax[row], plot_only=[1], style=style_file, linewidth=1, plot_loops=False)
            
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1]
            
            for k in range(1, n_cols):
                ax[row][k].set_title(f"{row}{k-1}", fontsize=5)
                loop_id = loop_idx_sorted[k-1]
                plot_edges_on_scatter(ax=ax[row][k],
                                 edge_idx=res["cycles"][1][loop_id],
                                 x=embd_n_mask,
                                 color=colors[k],
                                  linewidth=1)
                if k < geq_maxs[row]+1:
                    ax[row][0].scatter(*res["dgms"][1][loop_id].T, c=colors[k], s=10)
                else:
                    # ax[row][k].axis("off")
                    ax[row][k].remove()
            
            
            if geq_maxs[row]>0:
                ax[row][1].set_title(f"{full_dist}  \n {row}0 ", fontsize=5)
            ax[row][0].legend().set_visible(False)
            
            for n in range(1, n_cols):
                if dataset == "yao_smart_orange":
                    ax[row][n].set_xlim(-20, 40)
                    ax[row][n].set_ylim(5, 85)
                elif dataset == "yao_smart_purple":
                    ax[row][n].set_xlim(20, 80)
                    ax[row][n].set_ylim(-70, 40)
                elif "10x" in dataset and "orange" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-65, -15)
                    ax[row][n].set_ylim(25, 70)
                elif "10x" in dataset and "purple" in dataset and "_male" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-30, 30)
                elif "10x" in dataset and "orange" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-90, -40)
                    ax[row][n].set_ylim(-10, 50)
                elif "10x" in dataset and "purple" in dataset and "_female" in dataset:
                    ax[row][n].set_xlim(-40, 10)
                    ax[row][n].set_ylim(-90, -20)
            
            
            ax[row][0].set_xticks([])
            ax[row][0].set_yticks([])
            # ax[row][0].set_ylabel("Death")

            row+=1
    fig.suptitle(f"{dataset}, bottleneck, {method}, {scale}*median", fontsize = 12)       

    plt.show()
    if save is True:
        fig.savefig(os.path.join(fig_path, f"Bottleneck_median2_{dataset}_{method}.png"), dpi=300)
    elif type(save) == str:
        fig.savefig(os.path.join(fig_path, f"{save}.png"), dpi=300)
    
    return geq_maxs


def get_life_geq_std(all_res, dataset, method, ratio = "quarter", scale = 2):
    root_path = get_path("data")
    geq_maxs = []
    life_maxs = []
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            
            tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))
            if method != "split":
                bottleneck_dists = list(tryit.values())[0]
            elif method == "split":
                dig = int(dataset[-1])
                if "female" in dataset:
                    dig+=10
                # print("dig = ", dig)
                bottleneck_dists = np.hstack((list(tryit.values())[0][:dig], list(tryit.values())[0][dig+1:]))
            life_max = np.mean(bottleneck_dists) + int(scale)* np.std(bottleneck_dists)
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            life_maxs.append(life_max)
            geq_maxs.append(geq_max)
 
    return life_maxs, geq_maxs

def get_life_geq_metrics(all_res, dataset, method, ratio = "quarter", scale = 2, metrics = "mean"):
    root_path = get_path("data")
    geq_maxs = []
    life_maxs = []
    for i, distance in enumerate(list(all_res.keys())):
        for j, full_dist in enumerate(all_res[distance]):
            res = all_res[distance][full_dist]
            life_array = all_res[distance][full_dist]["dgms"][1]
            reference_life_long = life_array[:,1] - life_array[:,0]
            
            tryit = load_dict(os.path.join(root_path, dataset, f"bottleneck_dists_{method}_{ratio}_{full_dist}.pkl"))
            if method != "split":
                bottleneck_dists = list(tryit.values())[0]
            elif method == "split":
                dig = int(dataset[-1])
                if "female" in dataset:
                    dig+=10
                # print("dig = ", dig)
                bottleneck_dists = np.hstack((list(tryit.values())[0][:dig], list(tryit.values())[0][dig+1:]))
            if metrics == "mean":
                life_max = int(scale)* (np.mean(bottleneck_dists) + np.std(bottleneck_dists))
            elif metrics == "median":
                life_max = int(scale)* (np.median(bottleneck_dists) + np.std(bottleneck_dists))
            geq_max = np.sum(np.array(reference_life_long) >= life_max)
            life_maxs.append(life_max)
            geq_maxs.append(geq_max)
 
    return life_maxs, geq_maxs