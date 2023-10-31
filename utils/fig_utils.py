from io_utils import dist_kwargs_to_str
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import glasbey
from vis_utils.plot import plot_scatter
from persim import plot_diagrams
import os


# print names for each hyperparameter setting
full_dist_to_print = {
    "euclidean": "Euclidean",
    "correlation": "Correlation",
    "fermat_p_2": "Fermat $p=2$",
    "fermat_p_3": "Fermat $p=3$",
    "fermat_p_5": "Fermat $p=5$",
    "fermat_p_7": "Fermat $p=7$",
    "dtm_k_4_p_dtm_2_p_radius_2": r"DTM $k=4, p=2, \xi=2$",
    "dtm_k_4_p_dtm_2_p_radius_inf": r"DTM $k=4, p=2, \xi=\infty$",
    "dtm_k_4_p_dtm_inf_p_radius_2": r"DTM $k=4, p=\infty, \xi=2$",
    "dtm_k_4_p_dtm_2_p_radius_1": r"DTM $k=4, p=2, \xi=1$",
    "dtm_k_4_p_dtm_inf_p_radius_inf": r"DTM $k=4, p=\infty, \xi=\infty$",
    "dtm_k_15_p_dtm_2_p_radius_2": r"DTM $k=15, p=2, \xi=2$",
    "dtm_k_15_p_dtm_inf_p_radius_1": r"DTM $k=15, p=\infty, \xi=1$",
    "dtm_k_15_p_dtm_inf_p_radius_2": r"DTM $k=15, p=\infty, \xi=2$",
    "dtm_k_15_p_dtm_inf_p_radius_inf": r"DTM $k=15, p=\infty, \xi=\infty$",
    "dtm_k_15_p_dtm_2_p_radius_inf": r"DTM $k=15, p=2, \xi=\infty$",
    "dtm_k_100_p_dtm_2_p_radius_1": r"DTM $k=100, p=2, \xi=1$",
    "dtm_k_100_p_dtm_inf_p_radius_1": r"DTM $k=100, p=\infty, \xi=1$",
    "dtm_k_100_p_dtm_2_p_radius_2": r"DTM $k=100, p=2, \xi=2$",
    "dtm_k_100_p_dtm_inf_p_radius_2": r"DTM $k=100, p=\infty, \xi=2$",
    "core_k_15": "Core $k=15$",
    "core_k_100": "Core $k=100$",
    "sknn_dist_k_15": "Geodesics $k=15$",
    "sknn_dist_k_100": "Geodesics $k=100$",
    "tsne_perplexity_30": r"t-SNE graph $\rho=30$",
    "tsne_perplexity_200": r"t-SNE graph $\rho=200$",
    "tsne_perplexity_333": r"t-SNE graph $\rho=333$",
    "umap_k_100_use_rho_True_include_self_True": "UMAP graph $k=100$",
    "umap_k_999_use_rho_True_include_self_True": "UMAP graph $k=999$",
    "tsne_embd_perplexity_8_n_epochs_500_n_early_epochs_250_rescale_tsne_True": r"t-SNE $\rho=8$",
    "tsne_embd_perplexity_30_n_epochs_500_n_early_epochs_250_rescale_tsne_True": r"t-SNE $\rho=30$",
    "tsne_embd_perplexity_333_n_epochs_500_n_early_epochs_250_rescale_tsne_True": r"t-SNE $\rho=333$",
    "umap_embd_k_15_n_epochs_750_min_dist_0.1_metric_euclidean": "UMAP $k=15$",
    "umap_embd_k_100_n_epochs_750_min_dist_0.1_metric_euclidean": "UMAP $k=100$",
    "umap_embd_k_999_n_epochs_750_min_dist_0.1_metric_euclidean": "UMAP $k=999$",
    "eff_res_corrected_True_weighted_False_k_4_disconnect_True": "Effective\resistance $k=4$",
    "eff_res_corrected_True_weighted_False_k_15_disconnect_True": "Effective\nresistance $k=15$",
    "eff_res_corrected_True_weighted_False_k_100_disconnect_True": "Effective\nresistance $k=100$",
    "eff_res_corrected_True_weighted_True_k_15_disconnect_True": "Effective\nresistance\nweighted $k=15$",
    "eff_res_corrected_True_weighted_True_k_100_disconnect_True": "Effective\nresistance\nweighted $k=100$",
    "eff_res_corrected_False_weighted_True_k_15_disconnect_True": "Eff. res. weighted\nuncorrected $k=15$",
    "eff_res_corrected_False_weighted_True_k_100_disconnect_True": "Eff. res. weighted\nuncorrected $k=100$",
    "eff_res_corrected_True_weighted_False_k_15_disconnect_True_sqrt": "Eff. res. $k=15$ square root",
    "eff_res_corrected_True_weighted_False_k_100_disconnect_True_sqrt": "Eff. res. $k=100$ square root",
    "diffusion_k_100_t_8_kernel_sknn_include_self_False": "Diffusion\n$k=100, t=8$",
    "diffusion_k_100_t_64_kernel_sknn_include_self_False": "Diffusion\n$k=100, t=64$",
    "diffusion_k_15_t_2_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=2$",
    "diffusion_k_15_t_4_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=4$",
    "diffusion_k_15_t_8_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=8$",
    "diffusion_k_15_t_16_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=16$",
    "diffusion_k_15_t_32_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=32$",
    "diffusion_k_15_t_64_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=64$",
    "diffusion_k_15_t_128_kernel_sknn_include_self_False": "Diffusion\n$k=15, t=128$",
    "spectral_k_15_normalization_sym_n_evecs_2_weighted_False": "Lap. Eig. $k=15, d=2$, sym",
    "spectral_k_15_normalization_sym_n_evecs_10_weighted_False": "Lap. Eig. $k=15, d=10$, sym",
    "spectral_k_15_normalization_none_n_evecs_2_weighted_False": "Lap. Eig. $k=15, d=2$",
    "spectral_k_15_normalization_none_n_evecs_5_weighted_False": "Lap. Eig. $k=15, d=5$",
    "spectral_k_15_normalization_none_n_evecs_10_weighted_False": "Lap. Eig. $k=15, d=10$",
}

# print names for the datasets
dataset_to_print = {"toy_circle": "Circle",
                    "inter_circles": "Linked circles",
                    "eyeglasses": "Eyeglasses",
                    "toy_blob": "Blob",
                    "torus": "Torus",
                    "toy_sphere": "Sphere",
                    "mca_ss2": "Malaria",
                    "neurosphere_gopca_small": "Neurosphere",
                    "neurosphere": "Neurosphere",
                    "hippocampus_gopca_small": "Hippocampus",
                    "hippocampus": "Hippocampus",
                    "pallium_scVI_IPC": "Neural IPCs",
                    "pallium_scVI_IPC_small": "Neural IPCs",
                    "HeLa2_gopca": "HeLa2",
                    "HeLa2": "HeLa2",
                    "pancreas_gopca": "Pancreas",
                    "pancreas": "Pancreas",
                    }

# print names for the distances
dist_to_print = {
    "euclidean": "Euclidean",
    "correlation": "Correlation",
    "fermat": "Fermat",
    "dtm": "DTM",
    "core": "Core",
    "sknn_dist": "Geodesics",
    "tsne": "t-SNE graph",
    "umap": "UMAP graph",
    "tsne_embd": "t-SNE",
    "umap_embd": "UMAP",
    "eff_res": "Eff. resist.",
    "diffusion": "Diffusion",
    "spectral": "Lap. eig.",
}


# colors, order of method in all_distances affects which colors they get
all_distances = {
    "euclidean": [{}],
    "umap": [
        {"k": 100, "use_rho": True, "include_self": True},
        {"k": 999, "use_rho": True, "include_self": True},
    ],
    "fermat": [
               {"p": 2},
               {"p": 3},
               {"p": 5},
               {"p": 7}
               ],
    "dtm": [

            {"k": 4, "p_dtm": 2, "p_radius": 1},
            {"k": 100, "p_dtm": 2, "p_radius": 1},
            {"k": 15, "p_dtm": np.inf, "p_radius": 1},
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
    ],
    "eff_res": [
        {"corrected": True, "weighted": False, "k": 15, "disconnect": True},
        {"corrected": True, "weighted": False, "k": 100, "disconnect": True}
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

}

all_full_dists = {dist: [dist + dist_kwargs_to_str(distance_kwargs) for distance_kwargs in all_distances[dist]] for dist in all_distances}

block_lengths = [len(all_full_dists[dist]) for dist in all_full_dists]
del block_lengths[0]  # remove euclidean
colors = glasbey.create_block_palette(block_lengths,
                                      generated_color_lightness_bounds= (20, 60),
                                      generated_color_chroma_bounds = (20, 100),
                                      max_lightness_bend=40)
colors.insert(0, "k")  # add black for euclidean

full_dist_to_color = {full_dist: color for full_dist, color in zip([full_dists for dist in all_full_dists for full_dists in all_full_dists[dist]], colors)}


dist_to_color = {}
for dist_ind, dist in enumerate(all_distances):
    if dist == "euclidean":
        dist_to_color[dist] = "k"
        continue
    block_start = 1 + sum(block_lengths[:dist_ind-1])
    block_end = block_start + block_lengths[dist_ind-1]
    mid_ind = np.ceil((block_start + block_end) / 2).astype(int)
    dist_to_color[dist] = colors[mid_ind]


def plot_metrics(distances, sigmas, metric, best_metric=None, ncols=4, metric_name=None):
    # plot the metrics for difference distances including a panel that show the best results for all distances, not used for paper

    ncols = min(ncols, len(distances)+1-("euclidean" in distances))  # one panel for best methods, but none for euclidean
    nrows = np.ceil((len(distances)-1)/ncols).astype(int)

    # fig size appropriate for jupyter but not for the paper
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(35,5*nrows), constrained_layout=True)
    if nrows == 1:
        ax = ax.reshape(1, -1)

    block_lenghts = [len(v) for k, v in distances.items()]

    # create color palette and set black for euclidean
    if "euclidean" in distances:
        ind = list(distances.keys()).index("euclidean")
        del block_lenghts[ind]
        colors = glasbey.create_block_palette(block_lenghts)  # omit euclidean from the color palette
        colors.insert(ind, "k")  # add black for euclidean
    else:
        colors = glasbey.create_block_palette(block_lenghts)

    i = 0  # color index
    for j, dist in enumerate(distances):

        # plot euclidean in all panels
        if dist == "euclidean":
            for r in range(nrows):
                for c in range(ncols):
                    ax[r, c].plot(sigmas, metric[dist]["euclidean"].mean(1), c=colors[i], linestyle="dashed", label=dist)
                    ax[r, c].fill_between(sigmas,
                                       metric[dist]["euclidean"].mean(1)-metric[dist]["euclidean"].std(1),
                                       metric[dist]["euclidean"].mean(1)+metric[dist]["euclidean"].std(1),
                                       color=colors[i],
                                       alpha=0.2)
            i = i + 1
        else:
            for k in metric[dist]:
                current_metric = metric[dist][k]
                # plot the best ones in the second panel, selects the best AUC for each type of distances
                if best_metric is None:
                    plot_now = True
                else:
                    if best_metric[dist]["run"] == k:
                        plot_now = True
                    else:
                        plot_now = False
                if plot_now:
                    ax[0, 0].plot(sigmas, current_metric.mean(1), c=colors[i], label=k)
                    ax[0, 0].fill_between(sigmas,
                                   current_metric.mean(1)-current_metric.std(1),
                                   current_metric.mean(1)+current_metric.std(1),
                                   color=colors[i],
                                   alpha=0.2)
                    ax[0, 0].set_title("Best AUC for each distance type")
                    ax[0, 0].set_ylim(-0.1, 1.1)
                    ax[0, 0].legend(loc="lower left")

                # plot all configurations for one distance in the subsequent panels
                if dist != "euclidean":
                    crow = j // ncols
                    ccol = j % ncols

                    ax[crow, ccol].plot(sigmas, current_metric.mean(1), c=colors[i], label=k)
                    ax[crow, ccol].fill_between(sigmas,
                                   current_metric.mean(1)-current_metric.std(1),
                                   current_metric.mean(1)+current_metric.std(1),
                                   color=colors[i],
                                   alpha=0.2)
                    ax[crow, ccol].set_title(dist)
                    ax[crow, ccol].set_ylim(-0.1, 1.1)
                    ax[crow, ccol].legend(loc="lower left")

                i = i+1

        for cax in ax.flatten():
            cax.set_xlabel("sigma")
            if metric_name is not None:
                cax.set_ylabel(metric_name)
    return fig


def plot_many_dists(outlier_scores, sigmas, ylabel, fig_path, fig_title):
    """
    Plot the detection scores for all distances and all configurations.
    :param outlier_scores: dictionary with detection scores. First two hierarchies are distances and specific distance
     configurations. Rest is array with dimensions len(sigmas), len(seeds).
    :param sigmas: array with sigmas
    :param ylabel: label for y axis indicating the dimensionality of hole (loop / void) and their number
    :param fig_path: path to save the figure
    :param fig_title: title of the figure
    :return: None
    """
    nrows = 3
    ncols = 4
    letters = "abcdefghijklmnopqrstuvwxyz"

    fig, ax = plt.subplots(ncols=ncols,
                           nrows=nrows,
                           figsize=(5.5, 5.5 / 1.2))

    mean_eucl = outlier_scores["euclidean"]["euclidean"].mean(1)
    std_eucl = outlier_scores["euclidean"]["euclidean"].std(1)

    shift = 0
    for i, distance in enumerate(outlier_scores):
        if distance == "euclidean": continue
        if i == 9:
            shift = -1
        i -= 1 + shift
        c, r = divmod(i, nrows)

        for full_dist in outlier_scores[distance]:
            mean = outlier_scores[distance][full_dist].mean(1)
            std = outlier_scores[distance][full_dist].std(1)
            ax[r, c].plot(sigmas,
                          mean,
                          # c=full_dist_to_color[full_dist],
                          c=full_dist_to_color[full_dist],
                          label=full_dist_to_print[full_dist].replace("\n", " "),
                          # label= dist_to_print[distance],
                          clip_on=False
                          )
            ax[r, c].fill_between(sigmas,
                                  mean + std,
                                  mean - std,
                                  alpha=0.2,
                                  color=full_dist_to_color[full_dist],
                                  # color[distance],
                                  edgecolor=None
                                  )

        if i == 2:
            ax[r, c].plot(sigmas,
                          mean_eucl,
                          c="k",
                          linestyle="dashed",
                          # label=full_dist_to_print["euclidean"],
                          label=dist_to_print["euclidean"],
                          clip_on=False)
        else:
            ax[r, c].plot(sigmas, mean_eucl, c="k", linestyle="dashed", clip_on=False)

        ax[r, c].fill_between(sigmas,
                              mean_eucl + std_eucl,
                              mean_eucl - std_eucl,
                              color="k",
                              alpha=0.2,
                              edgecolor=None)

        ax[r, c].set_ylim(0, 1)
        ax[r, c].set_xlim(0, max(sigmas))
        ax[r, c].set_xlabel("Noise std $\sigma$")
        if i == 0:
            ax[r, c].set_ylabel(ylabel)

        if c > 0:
            ax[r, c].set_yticklabels([])
        else:
            ax[r, c].set_ylabel(ylabel)
        if r < nrows - 1:
            ax[r, c].set_xticklabels([])
            ax[r, c].set_xlabel("")
            leg_y = -0.1
        else:
            leg_y = -0.5

        ax[r, c].legend(loc='upper center',
                        bbox_to_anchor=(0.5, leg_y),
                        handlelength=1.,
                        frameon=False
                        )

        ax[r, c].set_title(dist_to_print[distance])
        ax[r, c].set_title(
            letters[i],
            loc="left",
            ha="right",
            fontweight="bold",
        )

    ax[2, 2].axis("off")

    fig.savefig(os.path.join(fig_path, fig_title))


def plot_dgm_loops(res,
                   embd,
                   y,
                   n_loops=4,
                   n_cols=5,
                   cmap="tab20",
                   plot_only=None,
                   s=1,
                   existing_colors=None,
                   confidence=None,
                   ax=None):
    """
    Plot the persistence diagram and the most persistent loops.
    :param res: ripser result dict
    :param embd: 2D or 3D embedding of the points
    :param y: Colors for the scatter plots or values for a colormap
    :param n_loops: number of loops to plot
    :param n_cols: Number of columns in the plot. Can be overwritten by 1+n_loops if the latter is higher and only one
    row is used.
    :param cmap:  colormap
    :param plot_only: None or list of the feature dimensions to plot in the persistence diagram
    :param s: Size of the scatter points
    :param existing_colors: List of colors already used in the plot. Used to avoid similar colors for the scatter plots
    and the loops.
    :param confidence: Size of the confidence interval. If a float, this will plot a shaded confidence band around the
    diagonal at that distance. If a list of two floats this plots two lines parallel to the diagonal at those distances.
    :param ax: Pass an axis to plot on. If None, a new figure is created. This is convenient if want to plot multiple
     rows of persistence diagram + representatives in one figure.
    :return: figure and axes (if no ax was passed) or just the updated axes (if ax was passed).
    """
    if existing_colors is None:
        existing_colors = []  # this makes it impossible to pass exisiting colors via y, but enables colored scatter plots
    if len(res["dgms"][1]) ==0:
        return ax

    # plot a warning if multiple points have maximal death time
    nb_max_death = (res["dgms"][1][:, 1] == res["dgms"][1].max()).sum()
    if nb_max_death > 1:
        print(f'Number of loops with maximal death time: {nb_max_death}')

    # sort loop life times
    life_times_loops = get_life_times(res, dim=1)
    loop_idx_sorted = np.argsort(life_times_loops)[::-1]

    if ax is None:
        # set up figure
        n_rows = np.ceil((n_loops+1) / n_cols).astype(int)
        if n_rows == 1:
            n_cols = n_loops+1

        # handle 2D and 3D dataset differently
        if embd.shape[1] == 2:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), layout="constrained")
        elif embd.shape[1] == 3:
            mosaic = np.array([[f"{i}{j}" for i in range(n_cols)] for j in range(n_rows)])
            mosaic_kw = {idx_str: {} if idx_str == "00" else {"projection": "3d"} for idx_str in mosaic.flatten()}
            fig, ax = plt.subplot_mosaic(mosaic, per_subplot_kw=mosaic_kw, figsize=(n_cols * 4, n_rows * 4))
            ax = np.array([[ax[idx_str] for idx_str in row] for row in mosaic])
            ax = ax.squeeze()
        fig_created = True
        ax0 = ax[0, 0] if n_rows > 1 else ax[0]
    else:
        # expects a single row of axes, where the first column shall hold a persistence diagram
        assert len(ax.shape) == 1
        ax0 = ax[0]
        n_rows = 1
        n_cols = len(ax)
        n_loops = n_cols - 1
        fig_created = False

    # create color palette, but avoid the colors used in the persistence diagram
    tab10 = matplotlib.cm.get_cmap("tab10")
    if plot_only is None:
        plot_only = list(res["dmgs"].keys())
    existing_colors = existing_colors + [tab10(i) for i in plot_only]
    colors = glasbey.extend_palette(existing_colors, n_loops+len(existing_colors)+2)[len(existing_colors)+1:]

    # plot persistence diagram
    plot_diagrams(res["dgms"], show=False, ax=ax0, size=5, plot_only=plot_only)

    # add confidence intervals
    if confidence is not None:
        if isinstance(confidence, float):
            ax0.fill_between(ax0.get_xlim(), ax0.get_xlim(), ax0.get_xlim()+confidence, alpha=0.05, color="r")
        elif isinstance(confidence, list):
            assert len(confidence) == 2
            ax0.fill_between(ax0.get_xlim(), ax0.get_xlim(), ax0.get_xlim()+confidence[0], alpha=0.05, color="r")
            ax0.plot(ax0.get_xlim(), ax0.get_xlim() + confidence[1], alpha=0.05, color="k", linestyle="--")


    # plot loops
    n_loops = np.min([n_loops, len(loop_idx_sorted)])  # in case there are not enough loops
    for i, loop_id in enumerate(loop_idx_sorted[:n_loops]):
        i = i + 1
        cax = ax[i // n_cols, i % n_cols] if n_rows > 1 else ax[i]

        # mark loop in persistence diagram
        ax0.scatter(*res["dgms"][1][loop_id].T, c=colors[i], s=10)

        # plot embd points
        plot_scatter(cax, embd, y, s=s, alpha=1, cmap=cmap, scalebar=False)

        # plot loop
        plot_edges_on_scatter(ax=cax,
                             edge_idx=res["cycles"][1][loop_id],
                             x=embd,
                             color=colors[i])

    if fig_created:
        return fig, ax
    else:
        return ax


def plot_edges_on_scatter(ax, edge_idx, x, color="k", linewidth=2, **kwargs):
    # plots a set of edges on top of a scatter plot in ax. Edges are given by the label of the vertices they connect.
    # The positions of the vertices are given in x.
    edges = np.moveaxis(np.stack([x[edge_idx[:, 0].astype(int)],
                                  x[edge_idx[:, 1].astype(int)]]),
                        0, 1)
    if x.shape[1] == 2:
        lc = mc.LineCollection(edges, color=color, linewidths=linewidth, zorder=6, **kwargs)
        lc.set_joinstyle("round")
        lc.set_capstyle("round")
        ax.add_collection(lc)
    elif x.shape[1] == 3:
        lc = Line3DCollection(edges, color=color, linewidths=linewidth, zorder=6, **kwargs)
        lc.set_joinstyle("round")
        lc.set_capstyle("round")
        ax.add_collection3d(lc)
        ax.view_init(elev=45, azim=-90)
        ax.axis("on")
    else:
        raise ValueError("Can only plot 2D and 3D embeddings.")
    return ax



