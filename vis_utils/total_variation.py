import numpy as np
import matplotlib.pyplot as plt
import pykeops
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.animation as animation
from matplotlib import collections  as mc
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform


from utils.utils import get_path
from utils.io_utils import load_multiple_res
from utils.pd_utils import get_life_times
from vis_utils.plot import plot_scatter
from utils.fig_utils import dataset_to_print, plot_dgm_loops, dist_to_print, plot_edges_on_scatter


def total_variation(dist_1, dist_2):
    return 0.5 * np.sum(np.abs(dist_1 - dist_2))


def reshape_dist(dist, num = None):
    """just reshape the frequency from list into a 2D array, for the purpose of visualization

    Args:
        dist (list): a list, containing the frequency across distances, that the most [num] persistant loops pass through each cell clusters
        num (int, optional): the number of persistant loops we want to keep in the . Defaults to None. If None, take everything from the dist, which is 7 loops per condition.

    Returns:
        ticks(list): a list of strings, representing the ticks on the x-axis, each string is a combination of the distance and the loop index. len(ticks) should be equal to the number of loops
        result(np.array): len(ticks) * number_of_cell_clusters,  a 2D array, representing the frequency of the loops passing through the clusters 
    """
    
    #num: number of loops per condition
    full_dist = list(dist.keys())[1:]
    size = len(dist[list(dist.keys())[0]])
    result = np.zeros((1,size))
    ticks = []
    # shorten_dist = ['euclidean',
    #     'diffusion_k_15_t_8',
    #     'diffusion_k_100_t_8',
    #     'diffusion_k_15_t_64',
    #     'diffusion_k_100_t_64',
    #     'eff_res_k_15',
    #     'eff_res_k_100']
    shorten_dist = ['euc',
        'diff_15_8',
        'diff_100_8',
        'diff_15_64',
        'diff_100_64',
        'eff_15',
        'eff_100']
    for shorten, condition in zip(shorten_dist, full_dist):
        if num is not None:
            result = np.vstack((result, np.array(dist[condition][:int(num)])))
            ticks.extend([f"{shorten}"+f"_{i}"for i in range(num)])
        else:
            result = np.vstack((result, np.array(dist[condition])))
            ticks.extend([f"{shorten}"+f"_{i}"for i in range(7)])
    return ticks, result[1:,:]


def complete_linkage(mat, threshold = 0.4):
    '''
    use the complete linkage to cluster the loops, and return the clustering index of each loop
    
    mat: total_variation_matrix, a 2D array, representing the total variation between two datasets
    
    return:
    clusters: a list, len(clusters) = number_of_loops, representing to clustering of with the threshold
    list_of_communities: contains same info as [clusters], just in a list, each element is a index of the loops with the same clustering index
    '''
    link = linkage(squareform(mat), method='complete')
    clusters = fcluster(link, t=threshold, criterion='distance')
    
    row_dendr = dendrogram(link)
    # print(pd.DataFrame(link, 
    #          columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
    #          index=['cluster %d' %(i+1) for i in range(link.shape[0])]))
    
    list_of_communities = [[] for _ in range(max(clusters))]
    for i, cluster in enumerate(clusters):
        list_of_communities[cluster-1].append(i)
    
    return clusters, list_of_communities

def total_variation_matrix(reshaped_array1, reshaped_array2):
    '''
    to calculate the total variation between two 2D arrays, if reshaped_array1 == reshaped_array2, then the total_variation of one dataset;
    if reshaped_array1 != reshaped_array2, then the total_variation of two datasets
    
    reshaped_array1, reshaped_array1: 2D array, representing the frequency of the loops passing through the clusters
    
    return: a 2D array, representing the total variation between two datasets
    '''
    size = reshaped_array1.shape[0]
    result = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            result[i][j] = total_variation(reshaped_array1[i,:], reshaped_array2[j,:])
    return result       
        
    
def pic_pigeonhole_loops(partition, dataset, embd_n_mask, distances, xlim = None, ylim = None, saveas = None):
    '''
    print out the loops with embedding. a row is a community. 
    
    partition: a dictionary, where the key is the index of the loops, and the value is the index of the community
    dataset: the name of the dataset
    embd_n_mask: the embedding of the dataset, AND the mask of the dataset(mask = we only take orange/purple cell groups)
    distances: THE DISTANCES YOU KNOW WHAT
    xlim, ylim: the x/y limits of the scatter plot, bc its cell group and dataset dependent
    saveas: a string, the name of the file you want to save the figure as, must be a *.png file
    '''
    
    if type(embd_n_mask) is tuple:
        embd_n_mask = embd_n_mask[0][embd_n_mask[1]]
    rows = len(set(partition.values()))
    columns = np.max([list(partition.values()).count(value) for value in range(rows)])
    
    width_ratios = [0.75] + [1] * (columns-1)
    fig, ax = plt.subplots(rows, columns, figsize=(columns*1, rows*1), constrained_layout=True)
    

    root_path = get_path("data")
    all_res = load_multiple_res(datasets=dataset, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)


    plot_loops = True
    colors = ['#d714fb', '#006100', '#5d0014', '#75ff3d', '#ffcef3', '#656979', '#db0c59']
    idx = 0
    pos_list = [0]*rows
    for distance in distances.keys():
        for full_dist in all_res[distance]:
            res = all_res[distance][full_dist]
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1][:7]
            for n in range(7):
                row = partition[idx]
                cax = ax[row][pos_list[row]]
                cax.set_title(idx, fontsize=7)
                if ylim is not None:
                    cax.set_ylim(*ylim)
                if xlim is not None:
                    cax.set_xlim(*xlim)
                plot_scatter(cax, embd_n_mask, y="k", s=1, alpha=1, scalebar=False)
                plot_edges_on_scatter(ax=cax,
                                 edge_idx=res["cycles"][1][loop_idx_sorted[n]],
                                 x=embd_n_mask,
                                 color=colors[n],
                                 linewidth=1)
                pos_list[row]+=1
                idx+=1
    for axx in ax.flat:
        axx.axis('off')
    fig.suptitle(dataset, fontsize = 16)       
    plt.show()
    if saveas is not None:
        fig.savefig(os.path.join(get_path("figures"), saveas), dpi=300)


def turn_to_partition(list_of_communities):
    """put list of communities into a dictionary.

    Args:
        list_of_communities : manually corrected list of communities

    Returns:
        manual_partition:  a distionary, where the key is the index of the loops, and the value is the index of the community
    """
    
    
    
    manual_partition = {num: idx for idx, sublist in enumerate(list_of_communities) for num in sublist}
    manual_partition = dict(sorted(manual_partition.items()))
    return manual_partition


# def complete_linkage_tri(threshold = 0.8):
#     size = mat_tasic.shape[0]
    
#     np.random.seed(0)
#     zeros = np.zeros(mat_y_10.shape)
#     tri_mat_raw = np.block([[mat_tasic, mat_t_y, mat_t_10],[zeros,mat_yao,mat_y_10],[zeros,zeros,mat_10x]])
#     upper_tri = np.triu(tri_mat_raw)
#     tri_mat = upper_tri + upper_tri.T
#     np.fill_diagonal(tri_mat, 0)
    
#     link = linkage(squareform(tri_mat), method='complete')
#     clusters = fcluster(link, t=threshold, criterion='distance')
    
#     row_dendr = dendrogram(link)
    
#     list_of_communities = [[] for _ in range(max(clusters))]
#     for i, cluster in enumerate(clusters):
#         list_of_communities[cluster-1].append(i)    
        
#     trim_list_of_communities = []
#     waste = []
#     for lst in list_of_communities:
#         if any(x <49 for x in lst) and any(49 <= x and x  < 2*49 for x in lst) and any(x >= 2*49 for x in lst):
#             trim_list_of_communities.append(lst)
#         else:
#             waste.append(lst)
#     trim_list_of_communities.append(waste)
    
#     partition = {}
#     part_trim_list_of_communities = trim_list_of_communities[:-1] + trim_list_of_communities[-1]
#     for index, lst in enumerate(part_trim_list_of_communities):
#         if isinstance(lst, list):
#             for number in lst:
#                 partition[number] = index  
#         else:
#             partition[lst] = index
    
#     return partition, trim_list_of_communities

def complete_linkage_tri(tri_mat, threshold = 0.8):
    link = linkage(squareform(tri_mat), method='complete')
    clusters = fcluster(link, t=threshold, criterion='distance')
    # row_dendr = dendrogram(link)
    
    untrim_list_of_communities = [[] for _ in range(max(clusters))]
    untrim_idx = [[] for _ in range(max(clusters))]
    untrim_data = [[] for _ in range(max(clusters))]
    for i, cluster in enumerate(clusters):
        untrim_list_of_communities[cluster-1].append(i)    
        untrim_data[cluster-1].append(data_tri[i])
        untrim_idx[cluster-1].append(idx_tri[i])
    
    trim_list_of_communities = []
    trim_data = []
    trim_idx = []
    waste = []
    waste_data = []
    waste_idx = []
    for data, idx, lst in zip(untrim_data, untrim_idx, untrim_list_of_communities):
        if len(set(data)) == 3:
            trim_list_of_communities.append(lst)
            trim_data.append(data)
            trim_idx.append(idx)
        else:
            waste.append(lst)
            waste_data.append(data)
            waste_idx.append(idx)
    trim_list_of_communities.append(waste)
    trim_data.append(waste_data)
    trim_idx.append(waste_idx)
    
    return  trim_list_of_communities, trim_data, trim_idx


# def tune_tri_threshold(lower_b = 0.5, higher_b = 1.0):
#     ts = np.arange(lower_b, higher_b, 0.01)
#     ave_tv = np.zeros((ts.shape))
#     n_tv = np.zeros((ts.shape))
#     size = 49
    
#     zeros = np.zeros(mat_y_10.shape)
#     tri_mat_raw = np.block([[mat_tasic, mat_t_y, mat_t_10],[zeros,mat_yao,mat_y_10],[zeros,zeros,mat_10x]])
#     upper_tri = np.triu(tri_mat_raw)
#     tri_mat = upper_tri + upper_tri.T
#     np.fill_diagonal(tri_mat, 0)
    
#     for i, t in enumerate(ts):
#         partition, trim_list_of_communities = complete_linkage_tri(threshold = t)
#         # print(len(trim_list_of_communities))
#         tvs = []
#         for l in trim_list_of_communities[:-1]:
#             l_tasic = [a for a in l if a <size]
#             l_yao = [b for b in l if b >=size and b < 2*size]
#             l_10x = [c for c in l if c >2*size]
#             # print(i)
#             # print([tri_mat[a,b], tri_mat[b,c], tri_mat[a,c]] for a in l_tasic for b in l_yao for c in l_10x)
#             tvs = tvs + [np.mean([tri_mat[a,b], tri_mat[b,c], tri_mat[a,c]]) for a in l_tasic for b in l_yao for c in l_10x]
#         ave_tv[i] = np.mean(np.array(tvs))
#         n_tv[i] = len(tvs)
#     return tri_mat, ave_tv, n_tv

def tune_tri_threshold(tri_mat, lower_b = 0.5, higher_b = 1.0):
    ts = np.arange(lower_b, higher_b, 0.01)
    ave_tv = np.zeros((ts.shape))
    n_tv = np.zeros((ts.shape))    
    
    for i, t in enumerate(ts):
        trim_list_of_communities, trim_data, trim_idx = complete_linkage_tri(tri_mat, threshold = t)
        tvs = []
        for idx, l in zip(trim_idx[:-1], trim_list_of_communities[:-1]):
            rows, cols = np.meshgrid(l, l)
            upper_idx = np.triu_indices(len(l), k=1)
            upper_lst= tri_mat[rows, cols][upper_idx].tolist()
            
            tvs = tvs + [np.mean(upper_lst)]
        ave_tv[i] = np.mean(np.array(tvs))
        n_tv[i] = len(tvs)
    return tri_mat, ave_tv, n_tv



def fig_tri_pigeonhole_loops(trim_list_of_communities, ticks, title, threshold = None, saveas = None):
    size = len(trim_list_of_communities)
    fig_list_of_communities = trim_list_of_communities.copy()
    # print(len(fig_list_of_communities[-1] ))
    fig_list_of_communities[-1] =  sorted(sum(fig_list_of_communities[-1], []))
    
    fig, ax = plt.subplots(size, 4, figsize=(8, size),  constrained_layout=True)
    ax[0][0].set_title("                                    loops in groups", fontsize=7)
    for i in range(size):
        loops_idx_tasic = [a for a in fig_list_of_communities[i] if a <49]
        loops_idx_yao =  [a-49 for a in fig_list_of_communities[i] if a >= 49 and a < 98]
        loops_idx_10x =  [a-98 for a in fig_list_of_communities[i] if a >= 98]
        
        sub_t_y = mat_t_y[np.ix_(loops_idx_tasic, loops_idx_yao)]
        sub_t_10 = mat_t_10[np.ix_(loops_idx_tasic, loops_idx_10x)]
        sub_y_10 = mat_y_10[np.ix_(loops_idx_yao, loops_idx_10x)]
        
        values, base = np.histogram(np.concatenate((sub_t_y.flatten(),sub_t_10.flatten(),sub_y_10.flatten())), bins=100) #???
        cumulative = np.cumsum(values)
        fig.suptitle(title)
        v_min = np.min(base)
        v_max = np.max(base)
        
        ax[i, 0].plot(base[:-1], cumulative/cumulative[-1], c='blue')
        ax[i, 0].set_xlim([0, 1])
        ax[i, 0].set_xlabel("Total variation")
        # ax[i, 0].set_ylabel("% of pairs of conditions")
        if threshold is not None:
            ax[i, 0].axvline(threshold[i], c="red")
            submatrix = np.where(submatrix > threshold[i], threshold[i], submatrix)
    
        # ax1 = ax[i,1].imshow(submatrix, cmap='grey')
        ax1 = ax[i,1].imshow(sub_t_y, cmap='grey', vmin = v_min, vmax = v_max)
        ytick1 = [f"{a}_{np.array(ticks)[a]}"for a in loops_idx_tasic]
        ax[i,1].set_yticks(range(len(loops_idx_tasic)), ytick1, size='small') #np.array(ticks)[loops_idx]
        ax[i,1].set_xticks(range(len(loops_idx_yao)), [a+49 for a in loops_idx_yao], size='small', rotation=90, ha='right')# np.array(ticks)[loops_idx]
        ax[i,1].set_ylabel("tasic")
        ax[i,1].set_xlabel("smart")
        
        ax2 = ax[i,2].imshow(sub_t_10, cmap='grey', vmin = v_min, vmax = v_max)
        ax[i,2].set_yticks(range(len(loops_idx_tasic)), ytick1, size='small') #np.array(ticks)[loops_idx]
        ax[i,2].set_xticks(range(len(loops_idx_10x)), [a+98 for a in loops_idx_10x], size='small', rotation=90, ha='right')# np.array(ticks)[loops_idx]
        ax[i,2].set_ylabel("tasic")
        ax[i,2].set_xlabel("10x")
        
        ax3 = ax[i,3].imshow(sub_y_10, cmap='grey', vmin = v_min, vmax = v_max)
        ytick2 = [f"{a+49}_{np.array(ticks)[a]}"for a in loops_idx_yao]
        ax[i,3].set_yticks(range(len(loops_idx_yao)), ytick2, size='small') #np.array(ticks)[loops_idx]
        ax[i,3].set_xticks(range(len(loops_idx_10x)), [a+98 for a in loops_idx_10x], size='small', rotation=90, ha='right')# np.array(ticks)[loops_idx]
        ax[i,3].set_ylabel("yao")
        ax[i,3].set_xlabel("10x")
        
        
        waht1 = fig.colorbar(ax1, ax=ax[i, 1])
        waht2 = fig.colorbar(ax2, ax=ax[i, 2])
        waht3 = fig.colorbar(ax3, ax=ax[i, 3])
    ax[size-1][0].set_title("                                    ungroupable loops", fontsize=7)
    plt.show()
    if saveas is not None:
        fig.savefig(os.path.join("/gpfs01/berens/user/hzhang/eff-ph/figures", saveas), dpi=300)


def pic_tri_unpigeonhole_loops(partition, trim_list_of_communities, ticks, dataset_tasic, dataset_yao, dataset_10x, cell_group, saveas = None):
    rows = len(trim_list_of_communities[-1])
    columns = np.max([len(i) for i in trim_list_of_communities[-1]])

    fig, ax = plt.subplots(rows, columns, figsize=(columns*1, rows*1), constrained_layout=True)
    
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
    root_path = get_path("data")
    all_res_tasic = load_multiple_res(datasets=dataset_tasic, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_tasic = {key: value for distance in all_res_tasic.values() for key, value in distance.items()}
    all_res_yao = load_multiple_res(datasets=dataset_yao, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_yao = {key: value for distance in all_res_yao.values() for key, value in distance.items()}
    all_res_10x = load_multiple_res(datasets=dataset_10x, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_10x = {key: value for distance in all_res_10x.values() for key, value in distance.items()}
     
    plot_loops = True
    colors = ['#d714fb', '#006100', '#5d0014', '#75ff3d', '#ffcef3', '#656979', '#db0c59']
    idxs = [i for l in trim_list_of_communities[-1] for i in l]
    idx = 0
    pos_list = [0]*rows
    while idx <= 3*49:
        if idx not in idxs:
            idx+=1
            pass
        else: 
            if idx<49 and idx in idxs:
                res = new_all_res_tasic[list(new_all_res_tasic.keys())[int(np.floor(idx/7))]]
                if cell_group == "orange":
                    embd_n_mask = embd_tasic[tasic_mask_orange]
                elif cell_group == "purple":
                    embd_n_mask = embd_tasic[tasic_mask_purple]
            elif idx >= 40 and idx < 98 and idx in idxs:
                res = new_all_res_yao[list(new_all_res_tasic.keys())[int(np.floor(idx/7))-7]]
                if cell_group == "orange":
                    embd_n_mask =  embd_yao[smart_mask_orange]
                elif cell_group == "purple":
                    embd_n_mask =  embd_yao[smart_mask_purple]
            elif idx >= 98 and idx in idxs:
                res = new_all_res_10x[list(new_all_res_tasic.keys())[int(np.floor(idx/7))-14]]
                if cell_group == "orange":
                    embd_n_mask = embd_orange_10x
                    # print(embd_n_mask.shape)
                elif cell_group == "purple":
                    embd_n_mask = embd_purple_10x
            life_times_loops = get_life_times(res, dim=1)
            loop_idx_sorted = np.argsort(life_times_loops)[::-1][:7]
            
            row = [i for i, l in enumerate(trim_list_of_communities[-1]) if idx in l][0]
            cax = ax[row][pos_list[row]]
            cax.set_title(idx, fontsize=7)
            if cell_group == "orange":
                if idx<49:
                    cax.set_ylim((30,80))
                elif idx >=40 and idx < 98:
                    cax.set_xlim((-85,0))
                    cax.set_ylim((-30,30))
                else:
                    cax.set_ylim((30, 90))
                    
            elif cell_group == "purple":
                if idx<49:
                    cax.set_ylim((-40,40))
                elif idx >=40 and idx < 98:
                    cax.set_ylim((0,80))
                else:
                    cax.set_xlim((-75, -60))
                    cax.set_ylim((-40, 40))
            plot_scatter(cax, embd_n_mask, y="k", s=1, alpha=1, scalebar=False)
            plot_edges_on_scatter(ax=cax,
                                edge_idx=res["cycles"][1][loop_idx_sorted[idx%7]],
                                x=embd_n_mask,
                                color=colors[idx%7],
                                linewidth=1)
            pos_list[row]+=1
            idx+=1
    for axx in ax.flat:
        axx.axis('off')
    # ax[10][0].set_title("                                    ungroupable loops", fontsize=30)
    fig.suptitle(f"tasic + smart seq + 10x, {cell_group}, ungroupable loops", fontsize = 30)       
    plt.show()
    if saveas is not None:
        fig.savefig(os.path.join("/gpfs01/berens/user/hzhang/eff-ph/figures", saveas), dpi=300)


# def pic_tri_pigeonhole_loops(partition, trim_list_of_communities, ticks, dataset_tasic, dataset_yao, dataset_10x, cell_group, saveas = None):
    
#     rows = len(trim_list_of_communities)-1
#     columns = np.max([list(partition.values()).count(value) for value in range(rows)])

#     fig, ax = plt.subplots(rows, columns, figsize=(columns*1, rows*1), constrained_layout=True)
    
#     distances = {
#         "euclidean": [{}],
#             "diffusion": [
#             {"k": 15, "t": 8, "kernel": "sknn", "include_self": False},
#             {"k": 100, "t": 8, "kernel": "sknn", "include_self": False},
#             {"k": 15, "t": 64, "kernel": "sknn", "include_self": False},
#             {"k": 100, "t": 64, "kernel": "sknn", "include_self": False},
#         ],
#         "eff_res": [
#             {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
#             {"corrected": True, "weighted": False, "k": 100, "disconnect": True},
#         ],
#     }
#     root_path = get_path("data")
#     all_res_tasic = load_multiple_res(datasets=dataset_tasic, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
#     new_all_res_tasic = {key: value for distance in all_res_tasic.values() for key, value in distance.items()}
#     all_res_yao = load_multiple_res(datasets=dataset_yao, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
#     new_all_res_yao = {key: value for distance in all_res_yao.values() for key, value in distance.items()}
#     all_res_10x = load_multiple_res(datasets=dataset_10x, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
#     new_all_res_10x = {key: value for distance in all_res_10x.values() for key, value in distance.items()}
    
    
#     plot_loops = True
#     colors = ['#d714fb', '#006100', '#5d0014', '#75ff3d', '#ffcef3', '#656979', '#db0c59']
#     idxs = [i for l in trim_list_of_communities[:-1] for i in l]
#     idx = 0
#     pos_list = [0]*rows
#     while idx <= 3*49:
#         if idx not in idxs:
#             idx+=1
#             pass
#         else: 
#             if idx<49 and idx in idxs:
#                 res = new_all_res_tasic[list(new_all_res_tasic.keys())[int(np.floor(idx/7))]]
#                 if cell_group == "orange":
#                     embd_n_mask = embd_tasic[tasic_mask_orange]
#                 elif cell_group == "purple":
#                     embd_n_mask = embd_tasic[tasic_mask_purple]
#             elif idx >= 40 and idx < 98 and idx in idxs:
#                 res = new_all_res_yao[list(new_all_res_tasic.keys())[int(np.floor(idx/7))-7]]
#                 if cell_group == "orange":
#                     embd_n_mask =  embd_yao[smart_mask_orange]
#                 elif cell_group == "purple":
#                     embd_n_mask =  embd_yao[smart_mask_purple]
#             elif idx >= 98 and idx in idxs:
#                 res = new_all_res_10x[list(new_all_res_tasic.keys())[int(np.floor(idx/7))-14]]
#                 if cell_group == "orange":
#                     embd_n_mask = embd_orange_10x
#                     # print(embd_n_mask.shape)
#                 elif cell_group == "purple":
#                     embd_n_mask = embd_purple_10x
#             life_times_loops = get_life_times(res, dim=1)
#             loop_idx_sorted = np.argsort(life_times_loops)[::-1][:7]
            
#             row = [i for i, l in enumerate(trim_list_of_communities[:-1]) if idx in l][0]
#             cax = ax[row][pos_list[row]]
#             cax.set_title(idx, fontsize=7)
#             if cell_group == "orange":
#                 if idx<49:
#                     cax.set_ylim((30,80))
#                 elif idx >=40 and idx < 98:
#                     cax.set_xlim((-85,0))
#                     cax.set_ylim((-30,30))
#                 else:
#                     cax.set_ylim((30, 90))
                    
#             elif cell_group == "purple":
#                 if idx<49:
#                     cax.set_ylim((-40,40))
#                 elif idx >=40 and idx < 98:
#                     cax.set_ylim((0,80))
#                 else:
#                     cax.set_xlim((-75, -60))
#                     cax.set_ylim((-40, 40))
#             plot_scatter(cax, embd_n_mask, y="k", s=1, alpha=1, scalebar=False)
#             plot_edges_on_scatter(ax=cax,
#                                 edge_idx=res["cycles"][1][loop_idx_sorted[idx%7]],
#                                 x=embd_n_mask,
#                                 color=colors[idx%7],
#                                 linewidth=1)
#             pos_list[row]+=1
#             idx+=1
#     for axx in ax.flat:
#         axx.axis('off')
#     # ax[10][0].set_title("                                    ungroupable loops", fontsize=30)
#     fig.suptitle(f"tasic + smart seq + 10x, {cell_group}, groupable loops", fontsize = 30)       
#     plt.show()
#     if saveas is not None:
#         fig.savefig(os.path.join(get_path("figures"), saveas), dpi=300)


def pic_tri_pigeonhole_loops(trim_list_of_communities, trim_data, trim_idx, dataset_tasic, dataset_yao, dataset_10x, cell_group, groupable = True, save = False):
    if groupable == True:
        rows = len(trim_data)-1
        columns = np.max([len(l) for l in trim_data[:-1]])
        trim_list_of_communities = trim_list_of_communities[:-1]
        group_label = "groupable"
    else:
        rows = len(trim_data[-1])
        columns = np.max([len(l) for l in trim_data[-1]])
        trim_list_of_communities = trim_list_of_communities[-1]
        trim_idx = trim_idx[-1]
        trim_data = trim_data[-1]
        group_label = "ungroupable"
        
    fig, ax = plt.subplots(rows, columns, figsize=(columns, rows), constrained_layout=True)
    
    distances = {
        # "euclidean": [{}],
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
    root_path = get_path("data")
    all_res_tasic = load_multiple_res(datasets=dataset_tasic, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_tasic = {key: value for distance in all_res_tasic.values() for key, value in distance.items()}
    all_res_yao = load_multiple_res(datasets=dataset_yao, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_yao = {key: value for distance in all_res_yao.values() for key, value in distance.items()}
    all_res_10x = load_multiple_res(datasets=dataset_10x, n=None, embd_dims=None, sigmas=None, distances=distances, seeds=0, root_path=root_path, n_threads=1)
    new_all_res_10x = {key: value for distance in all_res_10x.values() for key, value in distance.items()}
    
    
    plot_loops = True
    tab10 = matplotlib.colormaps.get_cmap("tab10")
    existing_colors = [tab10(1)]
    colors = glasbey.extend_palette(existing_colors, columns+len(existing_colors)+5)[len(existing_colors)+1:]
    print(len(colors))
    for i, lst in enumerate(trim_list_of_communities):
        for j, item in enumerate(lst):
            data_set = int(trim_data[i][j])
            dist = int(trim_idx[i][j][0])
            idx = int(trim_idx[i][j][1])
            # print(idx)
            cax = ax[i][j]
            cax.set_title(trim_idx[i][j], fontsize=7)
            if data_set == 0:
                res = new_all_res_tasic[list(new_all_res_tasic.keys())[dist]]
                if cell_group == "orange":
                    embd_n_mask = tasic_orange_enm
                elif cell_group == "purple":
                    embd_n_mask = tasic_purple_enm                    
            elif data_set == 1:
                res = new_all_res_yao[list(new_all_res_tasic.keys())[dist]]
                if cell_group == "orange":
                    embd_n_mask =  smart_orange_enm
                    cax.set_xlim(-20, 40)
                    cax.set_ylim(5, 85)
                elif cell_group == "purple":
                    embd_n_mask =  smart_purple_enm
                    cax.set_xlim(20, 80)
                    cax.set_ylim(-70, 40)
            elif data_set == 2:
                res = new_all_res_10x[list(new_all_res_tasic.keys())[dist]]
                if cell_group == "orange":
                    embd_n_mask = x10_orange_enm
                    cax.set_xlim(-65, -15)
                    cax.set_ylim(25, 70)
                elif cell_group == "purple":
                    embd_n_mask = x10_purple_enm
                    cax.set_xlim(-90, -40)
                    cax.set_ylim(-30, 30)
            life_times_loops = get_life_times(res, dim=1)
            
            loop_idx_sorted = np.argsort(life_times_loops)[::-1][idx]
            
            plot_scatter(cax, embd_n_mask, y="k", s=1, alpha=1, scalebar=False)
            plot_edges_on_scatter(ax=cax,
                                edge_idx=res["cycles"][1][loop_idx_sorted],
                                x=embd_n_mask,
                                color=colors[idx+1],
                                linewidth=1)
    for axx in ax.flat:
        axx.axis('off')
    fig.suptitle(f"tasic + smart seq + 10x, {cell_group}, {group_label} loops", fontsize =5)       
    plt.show()
    if save is True:
        fig.savefig(os.path.join(get_path("figures"), f"tv_3_bottleneck_{cell_group}_{group_label}.png"), dpi=300)
