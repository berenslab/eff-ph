import pandas as pd
import numpy as np
import os
import torchvision
import scipy.sparse
from .utils import kNN_graph, save_dict, load_dict
from sklearn.decomposition import PCA
import zipfile
from .treutlein_preprocess import preprocess as treut_preprocess
from .zfish_preprocess import preprocess as zfish_preprocess
from .rnaseqTools import geneSelection, sparseload, sparseload_sample
import urllib.request
import progressbar # added
import requests
import tarfile
import h5py
import scanpy as sc
import subprocess
import matplotlib.pyplot as plt
import time


def categorical2numeric(y, return_unique=False):
    y_unique = np.unique(y)
    y_numeric = np.zeros_like(y)
    for i, y_u in enumerate(y_unique):
        y_numeric[y==y_u] = i
    if return_unique:
        return y_numeric, y_unique
    else:
        return y_numeric


# from https://github.com/stat-ml/ncvis
# use their get_pendigits.py and download_pendigits.sh to obtain the dataset
def load_pendigits(root_path):
    files = ["pendigits/optdigits.tes",
             "pendigits/optdigits.tra"]
    loaded = [None] * 2
    for f in files:
        df = pd.read_csv(os.path.join(root_path,f), header=None)
        for i in range(2):
            if i == 0:
                new = df.iloc[:, :-1].values
            else:
                new = df.iloc[:, -1].values
            if loaded[i] is None:
                loaded[i] = new
            else:
                loaded[i] = np.concatenate((loaded[i], new))
    return loaded


def load_mnist(root_path):
    root_path = os.path.join(root_path, "mnist")
    mnist_train = torchvision.datasets.MNIST(root=root_path, train=True,
                                             download=True, transform=None)
    x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets
    mnist_test = torchvision.datasets.MNIST(root=root_path, train=False,
                                            download=True, transform=None)
    x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

    x = np.concatenate([x_train, x_test], axis=0)
    x = x.reshape(x.shape[0], -1)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y


def load_small_mnist(root_path, seed=0, size=6000):
    root_path = os.path.join(root_path, f"mnist_seed_{seed}_size_{size}")
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    try:
        x_small = np.load(os.path.join(root_path, "data.npy"))
        y_small = np.load(os.path.join(root_path, "labels.npy"))
    except FileNotFoundError:
        mnist_train = torchvision.datasets.MNIST(root=root_path, train=True,
                                                 download=True, transform=None)
        x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets
        mnist_test = torchvision.datasets.MNIST(root=root_path, train=False,
                                                download=True, transform=None)
        x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

        x = np.concatenate([x_train, x_test], axis=0)
        x = x.reshape(x.shape[0], -1)
        y = np.concatenate([y_train, y_test], axis=0)

        ind = []
        np.random.seed(seed)
        for i in range(10):
            ind_by_class = np.random.choice(np.argwhere(y == i).flatten(),
                                            size=int(size/ 10),
                                            replace=False)
            ind.extend(ind_by_class)
        perm = np.random.permutation(10 * int(size/10))
        ind = (np.array(ind)[perm],)
        x_small = x[ind, :][0]
        y_small = y[ind]

        np.save(os.path.join(root_path, "data.npy"),
                x_small)
        np.save(os.path.join(root_path, "labels.npy"),
                y_small)
    return x_small, y_small

def imbalance_dataset(x, y, props, seed=0):
    classes, counts = np.unique(y, return_counts= True)
    assert len(props) == len(np.unique(y))

    classes_idx = [ np.where(y == cls)[0] for cls in classes]

    new_counts = [int(prop * len(class_idx))
                        for prop, class_idx in zip(props, classes_idx) ]

    np.random.seed(seed)
    new_classes_idx = []
    for i in range(len(classes)):
        subsample_idx = np.random.permutation(counts[i])[:new_counts[i]]
        new_classes_idx.append(classes_idx[i][subsample_idx])
    full_subsample_idx = np.concatenate(new_classes_idx)

    return x[full_subsample_idx], y[full_subsample_idx]

def load_imba_mnist(root_path, props, seed=0):
    x, y = load_mnist(root_path)
    return imbalance_dataset(x, y, props, seed)

def load_cifar10(root_path):
    root_path = os.path.join(root_path, "cifar10")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    cifar10_train = torchvision.datasets.CIFAR10(root=root_path, train=True,
                                             download=True, transform=None)

    x_train, y_train = cifar10_train.data, cifar10_train.targets
    cifar10_test = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                            download=True, transform=None)
    x_test, y_test = cifar10_test.data, cifar10_test.targets

    x = np.concatenate([x_train, x_test], axis=0)
    x = x.reshape(x.shape[0], -1)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y

def load_human(root_path):
    root_path = os.path.join(root_path, "human-409b2")
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    try:
        x = np.load(os.path.join(root_path, "human-409b2.data.npy"))
        y = np.load(os.path.join(root_path, "human-409b2.labels.npy"))
        d = load_dict(os.path.join(root_path, "human-409b2.pkl"))
    except FileNotFoundError:
        #urls = ["https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.1.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.2.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.3.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.4.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.5.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.6.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.7.zip",]
        metafile = os.path.join(root_path,
                                #"unzipped_files",
                                "metadata_human_cells.tsv")
        countfile = os.path.join(root_path,
                                 #"unzipped_files",
                                 "human_cell_counts_consensus.mtx")
        urls = []
        if not os.path.exists(metafile):
            urls.append("http://ftp.ebi.ac.uk/biostudies/nfs/E-MTAB-/552/E-MTAB-7552/Files/metadata_human_cells.tsv")
        if not os.path.exists(countfile):
            urls.append("http://ftp.ebi.ac.uk/biostudies/nfs/E-MTAB-/552/E-MTAB-7552/Files/human_cell_counts_consensus.mtx")

        if len(urls) > 0:
            print("Downloading data")
        for url in urls:
            filename = os.path.join(root_path, url.split("/")[-1])
            #urllib.request.urlretrieve(url, filename)
            download_file(url, filename)
        #    print(filename)
        #    #with zipfile.ZipFile(filename, "r") as zip_ref:
        #    #    zip_ref.extractall(os.path.join(root_path, "unzipped_files"))
        #    with tarfile.open(filename, "r:gz") as f:
        #        f.extractall()
        #    assert False

        print("Preprocessing data")
        line = "409b2"
        X, stage = treut_preprocess(metafile, countfile, line)


        outputfile = "human-409b2"

        np.save(os.path.join(root_path, outputfile + ".data.npy"), X)
        np.save(os.path.join(root_path, outputfile + ".labels.npy"), stage)
        x = X
        y = stage

        print("Done")

        # meta data
        d = {"label_colors": {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }, "time_colors": {
            "  0 days": "navy",
            "  4 days": "royalblue",
            "10 days": "skyblue",
            "15 days": "lightgreen",
            "  1 month": "gold",
            "  2 months": "tomato",
            "  3 months": "firebrick",
            "  4 months": "maroon",
        }}

        # cluster assignments
        meta = pd.read_csv(metafile, sep="\t")
        mask = (meta["Line"] == "409b2")* meta["in_FullLineage"]

        d["clusters"] = list(meta[mask]["cl_FullLineage"])

        d["color_to_time"] = {v: k for k, v in d["time_colors"].items()}

        save_dict(d, os.path.join(root_path, f"{outputfile}.pkl"))

    d["clusters"] = np.array(d["clusters"])
    return x, y, d



# for translating labels to colors and time points
label_to_color = {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }

time_to_color = {
        "  0 days": "navy",
        "  4 days": "royalblue",
        "10 days": "skyblue",
        "15 days": "lightgreen",
        "  1 month": "gold",
        "  2 months": "tomato",
        "  3 months": "firebrick",
        "  4 months": "maroon",
    }

color_to_time = {v: k for k,v in time_to_color.items()}
label_to_time = {k: color_to_time[label_to_color[k]] for k in label_to_color.keys()}


def load_zebrafish(root_path):
    root_path = os.path.join(root_path, "zebrafish")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    try:
        x = np.load(os.path.join(root_path, "zfish.data.npy"))
        y = np.load(os.path.join(root_path, "zfish.labels.npy"))
    except FileNotFoundError:
        # download
        print("Downloading zebrafish data...")
        url = "https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/WagnerScience2018.h5ad"
        file_name = "WagnerScience2018.h5ad"
        file_path = os.path.join(root_path, file_name)
        urllib.request.urlretrieve(url, file_path)

        print("Preprocessing zebrafish data...")
        # preprocess
        X, stage, alt_c = zfish_preprocess(file_path)
        np.save(os.path.join(root_path, "zfish.data.npy"), X)
        np.save(os.path.join(root_path, "zfish.labels.npy"), stage)
        np.save(os.path.join(root_path, "zfish.altlabels.npy"), alt_c)
        print("...done.")

        x = X
        y = stage
    return x, y

zebra_label_to_color = {
    "4hpf": "navy",
    "6hpf": "royalblue",
    "8hpf": "skyblue",
    "10hpf": "lightgreen",
    "14hpf": "gold",
    "18hpf": "tomato",
    "24hpf": "firebrick",
    "unused": "maroon",
}

zebra_color_to_label = {val: key for key, val in zebra_label_to_color.items()}

def load_c_elegans(root_path):
    data_dir = os.path.join(root_path, "c_elegans")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir,
                                       "packer_c-elegans",
                                       "c-elegans_qc_final.txt")):
        # download the C. elegans data
        url = "http://cb.csail.mit.edu/cb/densvis/datasets/packer_c-elegans_data.tar.gz"
        file_name = os.path.join(data_dir, "packer_c-elegans_data.tar.gz")

        urllib.request.urlretrieve(url, file_name)

        # extract the data
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=data_dir)
        tar.close()


    x = pd.read_csv(os.path.join(data_dir,
                                 "packer_c-elegans",
                                 "c-elegans_qc_final.txt"),
                         sep='\t',
                         header=None)
    x = np.array(x)
    meta = pd.read_csv(os.path.join(data_dir,
                                    "packer_c-elegans",
                                    "c-elegans_qc_final_metadata.txt"),
                       sep=',',
                       header=0)

    cell_types = meta["cell.type"].to_numpy().astype(str)

    y = np.zeros(len(cell_types)).astype(int)
    #name_to_label = {}
    for i, phase in enumerate(np.unique(cell_types)):
        #name_to_label[phase] = i
        y[cell_types == phase] = i
    return x, y

def load_yao_smart_purple(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_smart_purple")
    try:
        x_purple = np.load(os.path.join(data_dir, "pca50.npy"))
        y_purple = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_smart(root_dataset)
        
        idx_purple = [i for i, cluster in enumerate(d["clusterNames"]) 
                if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        cluster_purple = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        
        mask_purple = [idx in idx_purple for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_purple]
        d["counts"] = d["counts"][mask_purple]
        # d["areas"] = d["areas"][mask_purple] #there is no area......

        x_purple = x[mask_purple]
        y_purple = y[mask_purple]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_purple)
        np.save(os.path.join(data_dir, "labels.npy"), y_purple)
    return x_purple, y_purple, d

def load_yao_smart_orange(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_smart_orange")
    try:
        x_orange = np.load(os.path.join(data_dir, "pca50.npy"))
        y_orange = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_smart(root_dataset)
        
        idx_orange = [i for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        cluster_orange = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        
        mask_orange = [idx in idx_orange for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_orange]
        d["counts"] = d["counts"][mask_orange]
        # d["areas"] = d["areas"][mask_orange] #there is no area......
        
        x_orange = x[mask_orange]
        y_orange = y[mask_orange]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_orange)
        np.save(os.path.join(data_dir, "labels.npy"), y_orange)
    
    return x_orange, y_orange, d

def load_yao_smart(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_smart")
    try:
        x = np.load(os.path.join(data_dir, "pca50.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        ##### 1. TO DOWNLOAD
        print("Downloading")
        urls = ["https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_smart-seq/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_smart-seq/metadata.csv"
                ]
        file_names = ["smart-seq_matrix.csv",
                      "smart-seq_metadata.csv"
                      ]      
        
        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloading {file_name}")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"...done. Time: {current_time}")

        
        ##### 2: GET counts
        print("Preprocessing")
        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        file_name_smart = os.path.join(data_dir, "smart-seq_matrix.csv")       

        counts, genes, cells = sparseload(file_name_smart, yao = True)
        
        print("counts are done....")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Preprocessing is done. Time: {current_time}")

        
        ##### 4: GET cluster, clusterNames, clusterColors
        print("Getting cluster thingy")
       
        clusterInfo = pd.read_csv(os.path.join(data_dir, "smart-seq_metadata.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_order'].values #in tasic was "cluster_id"
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values
        
        #to find the mapping between id and label and color. Changed: there is at least one cluster hasnt't got detected
        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids)) if len(labels[ids == i + 1])>0]) 
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids)) if len(colors[ids == i + 1])>0]) 

        ind = np.array([np.where(cells == c)[0][0] for c in goodCells if len(np.where(cells == c)[0])>0])
        counts = counts[ind, :]
        #this parts has not checked could go wrong in the function 
        unique_sorted = sorted(set(ids))
        value_map = {value: idx for idx, value in enumerate(unique_sorted)}
        clusters = np.array([value_map[num] for num in ids])
        #this parts has not checked could go wrong in the function 
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Getting cluster things is done. Time: {current_time}")
        
        yao = {'counts': counts, 'genes': genes, 'clusters': clusters,
                    'clusterColors': clusterColors, 'clusterNames': clusterNames}
        
        save_dict(yao, os.path.join(data_dir, "yao.pkl"))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving pkl thingy is done. Time: {current_time}")
        
        
        y = yao["clusters"]
        np.save(os.path.join(data_dir, "labels.npy"), y)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving labels.npy thingy is done. Time: {current_time}")


        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']
        
        
        importantGenesYao = geneSelection(
            yao['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=yao['genes'], plot=False)   #plot is false so the markers dont really matter....
        
        librarySizes = np.sum(yao_['counts'], axis=1)
     
        X = np.log1p(yao['counts'][:, importantGenesYao] / librarySizes * 1e+6) # / np.log(2)  # is supposed to be log_2 (1+x)
        X = np.asarray(X)
        X = X - X.mean(axis=0)
        U, s, V = np.linalg.svd(X, full_matrices=False)
        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        X = X[:, np.argsort(s)[::-1]][:, :50]

        x = X
        
        d = yao

        np.save(os.path.join(data_dir, "pca50.npy"), x)
        
        print("...done.")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"EVERYTHING is done. Time: {current_time}")

    return x, y, d

def find_mutual_genes(data_dir):
    file_10x = os.path.join(data_dir, "10x_matrix.csv")
    file_smart = os.path.join(data_dir, "smart-seq_matrix.csv")
    
    gene_smart = pd.read_csv(file_smart, nrows = 0).columns.tolist()
    gene_10x = pd.read_csv(file_10x, nrows = 0).columns.tolist()
    sane_columns = []
    for col in gene_10x:
        if col in gene_smart:
            sane_columns.append(col)
    
    np.save(os.path.join(data_dir, "mutual_genes.txt"), sane_columns)
    
    return sane_columns

def load_yao_10x_female_purple(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_female_purple")
    try:
        x_purple = np.load(os.path.join(data_dir, "pca50.npy"))
        y_purple = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_female(root_dataset)
        
        idx_purple = [i for i, cluster in enumerate(d["clusterNames"]) 
                if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        cluster_purple = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        
        mask_purple = [idx in idx_purple for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_purple]
        d["counts"] = d["counts"][mask_purple]
        d["cells"] = d["cells"][mask_purple]

        x_purple = x[mask_purple]
        y_purple = y[mask_purple]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_purple)
        np.save(os.path.join(data_dir, "labels.npy"), y_purple)
    return x_purple, y_purple, d

def load_yao_10x_female_orange(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_female_orange")
    try:
        x_orange = np.load(os.path.join(data_dir, "pca50.npy"))
        y_orange = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_female(root_dataset)
        
        idx_orange = [i for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        cluster_orange = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        
        mask_orange = [idx in idx_orange for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_orange]
        d["counts"] = d["counts"][mask_orange]
        d["cells"] = d["cells"][mask_orange]
        
        x_orange = x[mask_orange]
        y_orange = y[mask_orange]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_orange)
        np.save(os.path.join(data_dir, "labels.npy"), y_orange)
    
    return x_orange, y_orange, d

def load_yao_10x_male_green_56IT(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_male_green_56IT")
    try:
        x_green = np.load(os.path.join(data_dir, "pca50.npy"))
        y_green = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_male(root_dataset)
        
        idx_green = [i for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT" in cluster or "L5 IT" in cluster or "L5/6 IT" in cluster]
        cluster_green = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT" in cluster or "L5 IT" in cluster or "L5/6 IT" in cluster]
        
        mask_green = [idx in idx_green for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_green]
        d["counts"] = d["counts"][mask_green]
        # d["areas"] = d["areas"][mask_purple] #there is no area......

        x_green = x[mask_green]
        y_green = y[mask_green]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_green)
        np.save(os.path.join(data_dir, "labels.npy"), y_green)
    return x_green, y_green, d

def load_yao_10x_male_green_56IT2(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_male_green_56IT2")
    try:
        x_green = np.load(os.path.join(data_dir, "pca50.npy"))
        y_green = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_male(root_dataset)
        
        idx_green = [i for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT" in cluster or "L5 IT" in cluster]
        cluster_green = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT" in cluster or "L5 IT" in cluster]
        
        mask_green = [idx in idx_green for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_green]
        d["counts"] = d["counts"][mask_green]
        # d["areas"] = d["areas"][mask_purple] #there is no area......

        x_green = x[mask_green]
        y_green = y[mask_green]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_green)
        np.save(os.path.join(data_dir, "labels.npy"), y_green)
    return x_green, y_green, d


def load_yao_10x_male_green_56ITCTX(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_male_green_56ITCTX")
    try:
        x_green = np.load(os.path.join(data_dir, "pca50.npy"))
        y_green = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_male(root_dataset)
        
        idx_green = [i for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT CTX" in cluster or "L5 IT CTX" in cluster or "L5/6 IT CTX" in cluster]
        cluster_green = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                     if "L6 IT CTX" in cluster or "L5 IT CTX" in cluster or "L5/6 IT CTX" in cluster]
        
        mask_green = [idx in idx_green for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_green]
        d["counts"] = d["counts"][mask_green]
        # d["areas"] = d["areas"][mask_purple] #there is no area......

        x_green = x[mask_green]
        y_green = y[mask_green]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_green)
        np.save(os.path.join(data_dir, "labels.npy"), y_green)
    return x_green, y_green, d

def load_yao_10x_male_purple(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_male_purple")
    try:
        x_purple = np.load(os.path.join(data_dir, "pca50.npy"))
        y_purple = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_male(root_dataset)
        
        idx_purple = [i for i, cluster in enumerate(d["clusterNames"]) 
                if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        cluster_purple = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Lamp5" in cluster and "Lhx6" not in cluster or "_Vip" in cluster or "_Sncg" in cluster or "_Pax6" in cluster or "_Ntng1" in cluster]
        
        mask_purple = [idx in idx_purple for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_purple]
        d["counts"] = d["counts"][mask_purple]
        d["cells"] = d["cells"][mask_purple]

        x_purple = x[mask_purple]
        y_purple = y[mask_purple]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_purple)
        np.save(os.path.join(data_dir, "labels.npy"), y_purple)
    return x_purple, y_purple, d

def load_yao_10x_male_orange(root_dataset):
    data_dir = os.path.join(root_dataset, "yao_10x_male_orange")
    try:
        x_orange = np.load(os.path.join(data_dir, "pca50.npy"))
        y_orange = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        x, y, d = load_yao_10x_male(root_dataset)
        
        idx_orange = [i for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        cluster_orange = [cluster for i, cluster in enumerate(d["clusterNames"]) 
                    if "_Sst" in cluster or "_Pvalb" in cluster]
        
        mask_orange = [idx in idx_orange for idx in d['clusters']]

        d["clusters"] = d["clusters"][mask_orange]
        d["counts"] = d["counts"][mask_orange]
        d["cells"] = d["cells"][mask_orange]
        
        x_orange = x[mask_orange]
        y_orange = y[mask_orange]
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x_orange)
        np.save(os.path.join(data_dir, "labels.npy"), y_orange)
    
    return x_orange, y_orange, d

def load_yao_10x_male(root_dataset):
    #the only difference between the load_yao_10x_female is "10x_metadata_male.csv" or "10x_metadata_female.csv", and the result_dir
    data_dir = os.path.join(root_dataset, "yao")
    result_dir = os.path.join(root_dataset, "yao_10x_male")
    try:
        x = np.load(os.path.join(result_dir, "pca50.npy"))
        d = load_dict(os.path.join(result_dir, "yao.pkl"))
        y = d["clusters"]
        # y = np.load(os.path.join(result_dir, "labels.npy"))
        
    except FileNotFoundError:
        ##### 1. TO DOWNLOAD
        print("Downloading")       
        urls = [
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/metadata.csv",
                ]
        file_names = [
                      "10x_matrix.csv",
                      "10x_metadata.csv",
                      ]    
        
        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloading {file_name}")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"...done. Time: {current_time}")
        
    
        mutual_gene = os.path.join(data_dir, "mutual_genes.npy")
        if not os.path.exists(mutual_gene):
            sane_columns = find_mutual_genes(data_dir)
        else:
            sane_columns = np.load(mutual_gene)
            print("loading mutual genes...")
       
        
        ##### 2: GET counts
        print("Preprocessing")
        file_name_10x = os.path.join(data_dir, "10x_matrix.csv")      

        counts, genes, cells = sparseload_sample(file_name_10x, sane_columns, s = 50000, chunksize = 2500)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"counts are done. Time: {current_time}")
        print(counts.shape)
        
        importantGenesYao = geneSelection(
            counts, n=3000, threshold=32)
        
        sampled_columns = np.array(sane_columns[1:])[importantGenesYao.tolist()]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"selecting genes are done. Time: {current_time}")

        counts_full, genes_full, cells_full = sparseload(file_name_10x,  np.concatenate([["sample_name"],sampled_columns]), chunksize = 25000, yao = True)  #1169320

        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        what = {'counts': counts_full, 'genes': genes_full, 'cells_full': cells_full}
        save_dict(what, os.path.join(result_dir, "what.pkl"))
        
        print(f"Preprocessing is done. Time: {current_time}")
        ##### 4: GET cluster, clusterNames, clusterColors
        print("Getting cluster thingy")
        try:
            clusterInfo = pd.read_csv(os.path.join(data_dir, "10x_metadata_male.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])
        except:
            metadata = pd.read_csv(os.path.join(data_dir, "10x_metadata.csv"))
            male = metadata[metadata['donor_sex_id'] == 2]
            male.to_csv(os.path.join(data_dir, "10x_metadata_male.csv"), index = False)
            clusterInfo = pd.read_csv(os.path.join(data_dir, "10x_metadata_male.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_order'].values #in tasic was "cluster_id"
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values
        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids)) if len(labels[ids == i + 1])>0]) 
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids)) if len(colors[ids == i + 1])>0]) 

        unique_sorted = sorted(set(ids))
        value_map = {value: idx for idx, value in enumerate(unique_sorted)}
        ind = np.array([np.where(cells_full == c)[0][0] for c in goodCells if len(np.where(cells_full == c)[0])>0])
        print(ind[:100])
        counts_full = counts_full[ind, :]
        cells_full = cells_full[ind]
        clusters = np.array([value_map[num] for num in ids])
        
        np.save(os.path.join(result_dir, "labels.npy"), clusters)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Getting cluster things is done. Time: {current_time}")
        
        yao = {'counts': counts_full, 'genes': genes_full, 'cells':cells_full, 'clusters': clusters,
                    'clusterColors': clusterColors, 'clusterNames': clusterNames}
        
        save_dict(yao, os.path.join(result_dir, "yao.pkl"))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving pkl thingy is done. Time: {current_time}")      
       
        # yao = load_dict(os.path.join(result_dir, "yao.pkl"))
        
        librarySizes = np.sum(yao['counts'], axis=1)
        # [:, importantGenesYao]
        X = np.log1p(yao['counts'] / librarySizes * 1e+6) # / np.log(2)  # is supposed to be log_2 (1+x)
        X = np.asarray(X)
        # X_mean = X.mean(axis=0)
        # X = X - X_mean
        U, s, V = np.linalg.svd(X-X.mean(axis=0), full_matrices=False)

        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        x = X[:, np.argsort(s)[::-1]][:, :50]
        d = yao
        y = yao["clusters"]

        np.save(os.path.join(result_dir, "pca50.npy"), x)
        
        print("...done.")
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"EVERYTHING is done. Time: {current_time}")
    return x, y, d


def load_yao_10x_female(root_dataset):
    #the only difference between the load_yao_10x_male is "10x_metadata_male.csv" or "10x_metadata_female.csv", and the result_dir
    data_dir = os.path.join(root_dataset, "yao")
    result_dir = os.path.join(root_dataset, "yao_10x_female")
    try:
        x = np.load(os.path.join(result_dir, "pca50.npy"))
        d = load_dict(os.path.join(result_dir, "yao.pkl"))
        y = d["clusters"]
        # y = np.load(os.path.join(result_dir, "labels.npy"))
    except FileNotFoundError:
        ##### 1. TO DOWNLOAD
        print("Downloading")       
        urls = [
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/metadata.csv",
                ]
        file_names = [
                      "10x_matrix.csv",
                      "10x_metadata.csv",
                      ]    
        
        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloading {file_name}")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"...done. Time: {current_time}")
        
    
        mutual_gene = os.path.join(data_dir, "mutual_genes.npy")
        if not os.path.exists(mutual_gene):
            sane_columns = find_mutual_genes(data_dir)
        else:
            sane_columns = np.load(mutual_gene)
            print("loading mutual genes...")
       
        
        ##### 2: GET counts
        print("Preprocessing")
        file_name_10x = os.path.join(data_dir, "10x_matrix.csv")      

        counts, genes, cells = sparseload_sample(file_name_10x, sane_columns, s = 50000, chunksize = 2500)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"counts are done. Time: {current_time}")
        print(counts.shape)
        
        importantGenesYao = geneSelection(
            counts, n=3000, threshold=32)
        
        sampled_columns = np.array(sane_columns[1:])[importantGenesYao.tolist()]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"selecting genes are done. Time: {current_time}")

        counts_full, genes_full, cells_full = sparseload(file_name_10x,  np.concatenate([["sample_name"],sampled_columns]), chunksize = 25000, yao = True)  #1169320

        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        what = {'counts': counts_full, 'genes': genes_full, 'cells_full': cells_full}
        save_dict(what, os.path.join(result_dir, "what.pkl"))
        
        print(f"Preprocessing is done. Time: {current_time}")
        ##### 4: GET cluster, clusterNames, clusterColors
        print("Getting cluster thingy")
        try:
            clusterInfo = pd.read_csv(os.path.join(data_dir, "10x_metadata_female.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])
        except:
            metadata = pd.read_csv(os.path.join(data_dir, "10x_metadata.csv"))
            female = metadata[metadata['donor_sex_id'] == 1]
            female.to_csv(os.path.join(data_dir, "10x_metadata_female.csv"), index = False)
            clusterInfo = pd.read_csv(os.path.join(data_dir, "10x_metadata_female.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_order'].values #in tasic was "cluster_id"
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values
        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids)) if len(labels[ids == i + 1])>0]) 
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids)) if len(colors[ids == i + 1])>0]) 

        unique_sorted = sorted(set(ids))
        value_map = {value: idx for idx, value in enumerate(unique_sorted)}
        ind = np.array([np.where(cells_full == c)[0][0] for c in goodCells if len(np.where(cells_full == c)[0])>0])
        print(ind[:100])
        counts_full = counts_full[ind, :]
        cells_full = cells_full[ind]
        clusters = np.array([value_map[num] for num in ids])
        
        np.save(os.path.join(result_dir, "labels.npy"), clusters)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Getting cluster things is done. Time: {current_time}")
        
        yao = {'counts': counts_full, 'genes': genes_full, 'cells':cells_full, 'clusters': clusters,
                    'clusterColors': clusterColors, 'clusterNames': clusterNames}
        
        save_dict(yao, os.path.join(result_dir, "yao.pkl"))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving pkl thingy is done. Time: {current_time}")      
       
        yao = load_dict(os.path.join(result_dir, "yao.pkl"))
        
        print("loading yao from before")
        librarySizes = np.sum(yao['counts'], axis=1)
        # [:, importantGenesYao]
        X = np.log1p(yao['counts'] / librarySizes * 1e+6) # / np.log(2)  # is supposed to be log_2 (1+x)
        X = X.toarray()
        # X_mean = X.mean(axis=0)
        # X = X - X_mean
        U, s, V = np.linalg.svd(X-X.mean(axis=0), full_matrices=False)

        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        x = X[:, np.argsort(s)[::-1]][:, :50]
        d = yao
        y = yao["clusters"]

        np.save(os.path.join(result_dir, "pca50.npy"), x)
        
        print("...done.")
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"EVERYTHING is done. Time: {current_time}")
    return x, y, d

def load_yao_10x(root_dataset):
    data_dir = os.path.join(root_dataset, "yao")
    result_dir = os.path.join(root_dataset, "yao_10x")
    try:
        x = np.load(os.path.join(result_dir, "pca50.npy"))
        d = load_dict(os.path.join(result_dir, "yao.pkl"))
        y = d["clusters"]
        # y = np.load(os.path.join(result_dir, "labels.npy"))
        
    except FileNotFoundError:
        ##### 1. TO DOWNLOAD
        print("Downloading")       
        urls = [
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/metadata.csv",
                ]
        file_names = [
                      "10x_matrix.csv",
                      "10x_metadata.csv",
                      ]    
        
        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloading {file_name}")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"...done. Time: {current_time}")
        
    
        mutual_gene = os.path.join(data_dir, "mutual_genes.npy")
        if not os.path.exists(mutual_gene):
            sane_columns = find_mutual_genes(data_dir)
        else:
            sane_columns = np.load(mutual_gene)
            print("loading mutual genes...")
       
        
        ##### 2: GET counts
        print("Preprocessing")
        file_name_10x = os.path.join(data_dir, "10x_matrix.csv")      
        # file_name_10x_try = os.path.join(data_dir, "10x_matrix_10k.csv")  

        counts, genes, cells = sparseload_sample(file_name_10x, sane_columns, s = 50000, chunksize = 2500)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"counts are done. Time: {current_time}")
        print(counts.shape)
        
        importantGenesYao = geneSelection(
            counts, n=3000, threshold=32)
        
        sampled_columns = np.array(sane_columns[1:])[importantGenesYao.tolist()]
        print(sampled_columns)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"selecting genes are done. Time: {current_time}")

        # counts_full, genes_full, cells_full = sparseload_sample(file_name_10x, sampled_columns, s = 500000, chunksize = 10000, index_col=None)

        counts_full, genes_full, cells_full = sparseload(file_name_10x,  np.concatenate([["sample_name"],sampled_columns]), chunksize = 25000, yao = True)  #1169320
        # np.save(os.path.join(result_dir, "counts_full.npy"), counts_full) 
        # actually its quite good, tho only 367 genes are selected....... and 10 minutes in total.....
        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        what = {'counts': counts_full, 'genes': genes_full, 'cells_full': cells_full}
        save_dict(what, os.path.join(result_dir, "what.pkl"))
        
        print(f"Preprocessing is done. Time: {current_time}")
        ##### 4: GET cluster, clusterNames, clusterColors
        print("Getting cluster thingy")
       
        clusterInfo = pd.read_csv(os.path.join(data_dir, "10x_metadata.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_order'].values #in tasic was "cluster_id"
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values
        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids)) if len(labels[ids == i + 1])>0]) 
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids)) if len(colors[ids == i + 1])>0]) 

        unique_sorted = sorted(set(ids))
        value_map = {value: idx for idx, value in enumerate(unique_sorted)}
        ind = np.array([np.where(cells_full == c)[0][0] for c in goodCells if len(np.where(cells_full == c)[0])>0])
        print(ind[:100])
        counts_full = counts_full[ind, :]
        cells_full = cells_full[ind]
        clusters = np.array([value_map[num] for num in ids])
        
        np.save(os.path.join(result_dir, "labels.npy"), clusters)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Getting cluster things is done. Time: {current_time}")
        
        yao = {'counts': counts_full, 'genes': genes_full,'clusters': clusters,
                    'clusterColors': clusterColors, 'clusterNames': clusterNames}
        
        save_dict(yao, os.path.join(result_dir, "yao.pkl"))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving pkl thingy is done. Time: {current_time}")      
       
        # yao = load_dict(os.path.join(result_dir, "yao.pkl"))
        
        librarySizes = np.sum(yao['counts'], axis=1)
        # [:, importantGenesYao]
        X = np.log1p(yao['counts'] / librarySizes * 1e+6) # / np.log(2)  # is supposed to be log_2 (1+x)
        X = np.asarray(X)
        # X_mean = X.mean(axis=0)
        # X = X - X_mean
        U, s, V = np.linalg.svd(X-X.mean(axis=0), full_matrices=False)

        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        x = X[:, np.argsort(s)[::-1]][:, :50]
        
        # #larger data.....
        # # largedata = (cell_number, 3000)
        # largedata = largedata - X_mean
        # large_X = np.dot(largedata, V)[:, np.argsort(s)[::-1]] #????
        # #larger data.....
        
        d = yao
        y = yao["clusters"]

        np.save(os.path.join(result_dir, "pca50.npy"), x)
        
        print("...done.")
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"EVERYTHING is done. Time: {current_time}")
    return x, y, d


def load_yao(root_dataset):
    data_dir = os.path.join(root_dataset, "yao")
    try:
        x = np.load(os.path.join(data_dir, "pca50.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        ##### 1. TO DOWNLOAD
        print("Downloading")
        urls = [
            # "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985", 
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/metadata.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_smart-seq/matrix.csv",
                "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_smart-seq/metadata.csv"
                ]
        file_names = [
                    #   "mouse_VISp_gene_expression_matrices.zip",
                      "10x_matrix.csv",
                      "10x_metadata.csv",
                      "smart-seq_matrix.csv",
                      "smart-seq_metadata.csv"
                      ]      
        
        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloading {file_name}")
            # if file_name.endswith("_matrix.csv"):
            #     transposed_file = file_name[:-4]+ "_t.csv"
            #     print(transposed_file)
            #     if not os.path.exists(transposed_file):
            #         transpose_large_csv(file_name, chunksize = 10000)
            #         print(f"Transposing {transposed_file}")
            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, file_name.split("/")[-1].strip(".zip")))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"...done. Time: {current_time}")

        
        ##### 2: GET counts
        print("Preprocessing")
        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        
        # file_name_10x = os.path.join(data_dir, "10x_matrix_1k.csv")
        # file_name_smart = os.path.join(data_dir, "smart-seq_matrix_1k.csv") 
        
        file_name_10x = os.path.join(data_dir, "10x_matrix.csv")
        file_name_smart = os.path.join(data_dir, "smart-seq_matrix.csv")       

        # counts1, genes1, cells1 = sparseload(file_name_10x, yao = True)
        counts2, genes2, cells2 = sparseload(file_name_smart, yao = True)
        
        print("coutns are done....")
        sane_columns = []
        idx = []
        for i, col in enumerate(genes1):
            if col in genes2:
                sane_columns.append(col)
                idx.append(i)
        
        
        counts = scipy.sparse.vstack((newcounts1, newcounts2), format='csc')
        cells = np.concatenate((cells1, cells2))
        genes = genes1
        # counts = counts1
        # cells = cells1
        # genes = np.copy(genes1)
        # print("here i wanna check again if genes are correct.....")
        # if np.all(genes1==genes2):
        #     print("genes are correct!!!!")
        #     genes = np.copy(genes1)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Preprocessing is done. Time: {current_time}")
        
        # breakpoint()
        
        ##### 4: GET cluster, clusterNames, clusterColors
        print("Getting cluster thingy")
        clusterInfo_10x = pd.read_csv(os.path.join(data_dir, "10x_metadata.csv"),
                                      usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])
        clusterInfo_smart = pd.read_csv(os.path.join(data_dir, "smart-seq_metadata.csv"), 
                                        usecols=['sample_name', 'cluster_order',"cluster_label", "cluster_color"])

        clusterInfo = pd.concat([clusterInfo_10x, clusterInfo_smart], axis=0, ignore_index=True)

        # is it necessary to correct the cluster labels in the meta data???????? since its stored as '1_CR', '2_Meis2', not detailed at all...
        # wrong_to_correct = {'L6 CT ALM Nxph2 Sla': 'L6 CT Nxph2 Sla',
        #                     'L6b VISp Col8a1 Rprm': 'L6b Col8a1 Rprm',
        #                     'Sst Crh 4930553C11Rik ': 'Sst Crh 4930553C11Rik',
        #                     'Sst Myh8 Etv1 ': 'Sst Myh8 Etv1'}
        # for wrong, correct in wrong_to_correct.items():
        #     clusterInfo["cluster_label"] = clusterInfo["cluster_label"].replace(wrong, correct)

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_order'].values #in tasic was "cluster_id"
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values
        
        #to find the mapping between id and label and color. Changed: there is at least one cluster hasnt't got detected
        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids)) if len(labels[ids == i + 1])>0]) 
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids)) if len(colors[ids == i + 1])>0]) 
        clusters = np.copy(ids)
        tasoc = {}
        # print(goodCells[:100])
        # print(cells[:100])
        ind = np.array([np.where(cells == c)[0][0] for c in goodCells if len(np.where(cells == c)[0])>0])
        counts = counts[ind, :]
        # print(counts)

        areas = (ind < cells1.size).astype(int) #1 if from first file, 0 otherwise...

        clusters = clusters - 1
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Getting cluster things is done. Time: {current_time}")
        
        yao = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                    'clusterColors': clusterColors, 'clusterNames': clusterNames}
        
        save_dict(yao, os.path.join(data_dir, "yao.pkl"))
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving pkl thingy is done. Time: {current_time}")
        
        
        y = yao["clusters"]
        np.save(os.path.join(data_dir, "labels.npy"), y)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Saving labels.npy thingy is done. Time: {current_time}")


        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']
        
        
        importantGenesYao = geneSelection(
            yao['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=yao['genes'], plot=False)   #plot is false so the markers dont really matter....
        
        librarySizes = np.sum(yao['counts'], axis=1)

        #X = np.log2(yao['counts'][:, importantGenesYao] / librarySizes * 1e+6 + 1)       
        X = np.log1p(yao['counts'][:, importantGenesYao] / librarySizes * 1e+6) # / np.log(2)  # is supposed to be log_2 (1+x)
        X = np.asarray(X)
        # X = (yao['counts'][:, importantGenesYao] / librarySizes * 1e+6).log1p() # / np.log(2)  # is supposed to be log_2 (1+x)
        # X = X.toarray()  # bc np.array() does not work with sparse matrices
        X = X - X.mean(axis=0)
        U, s, V = np.linalg.svd(X, full_matrices=False)
        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        X = X[:, np.argsort(s)[::-1]][:, :50]

        x = X
        
        d = yao

        np.save(os.path.join(data_dir, "pca50.npy"), x)
        
        print("...done.")
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"EVERYTHING is done. Time: {current_time}")

    return x, y, d

def load_tasic(root_dataset):
    data_dir = os.path.join(root_dataset, "tasic")

    try:
        x = np.load(os.path.join(data_dir, "pca50.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "tasic2018.pkl"))
    except FileNotFoundError:

        print("Downloading")
        urls = ["http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985",
                "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179",
                "https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/raw/allen/tasic2018/sample_heatmap_plot_data.csv"
                ]

        file_names = ["mouse_VISp_gene_expression_matrices_2018-06-14.zip",
                      "mouse_ALM_gene_expression_matrices_2018-06-14.zip",
                      "sample_heatmap_plot_data.csv"
                      ]

        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)

            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, file_name.split("/")[-1].strip(".zip")))
        print("...done")

        print("Preprocessing")
        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        file_name_VISp = os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv")
        counts1, genes1, cells1 = sparseload(file_name_VISp)

        file_name_ALM = os.path.join(data_dir, "mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv")
        counts2, genes2, cells2 = sparseload(file_name_ALM)

        counts = scipy.sparse.vstack((counts1, counts2), format='csc')

        cells = np.concatenate((cells1, cells2))

        if np.all(genes1==genes2):
            genes = np.copy(genes1)

        genesDF = pd.read_csv(os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14",
                                           "mouse_VISp_2018-06-14_genes-rows.csv"))
        ids = genesDF['gene_entrez_id'].tolist()
        symbols = genesDF['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv(os.path.join(data_dir, "sample_heatmap_plot_data.csv"))

        # correct the cluster labels in the meta data
        wrong_to_correct = {'L6 CT ALM Nxph2 Sla': 'L6 CT Nxph2 Sla',
                            'L6b VISp Col8a1 Rprm': 'L6b Col8a1 Rprm',
                            'Sst Crh 4930553C11Rik ': 'Sst Crh 4930553C11Rik',
                            'Sst Myh8 Etv1 ': 'Sst Myh8 Etv1'}

        for wrong, correct in wrong_to_correct.items():
            clusterInfo["cluster_label"] = clusterInfo["cluster_label"].replace(wrong, correct)

        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)


        ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
        counts = counts[ind, :]

        areas = (ind < cells1.size).astype(int)

        clusters = clusters - 1

        tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                     'clusterColors': clusterColors, 'clusterNames': clusterNames}

        try:  #todo remove dependence on the clusterInfo file
            cell_type_meta = pd.read_excel(os.path.join(data_dir, "41586_2018_654_MOESM3_ESM", "Supplementary_Table_9_Cell_types_markers.xlsx"))
            full_meta = pd.read_excel(os.path.join(data_dir, "41586_2018_654_MOESM3_ESM", "Supplementary_Table_10_Full_Metadata.xlsx"))

            ############################################################################
            # corrections
            ############################################################################
            # correct the cluster labels in the full meta data
            wrong_to_correct = {'L6 CT ALM Nxph2 Sla': 'L6 CT Nxph2 Sla',
                                'L6b VISp Col8a1 Rprm': 'L6b Col8a1 Rprm',
                                'Sst Crh 4930553C11Rik ': 'Sst Crh 4930553C11Rik',
                                'Sst Myh8 Etv1 ': 'Sst Myh8 Etv1'}

            for wrong, correct in wrong_to_correct.items():
                full_meta["cluster"] = full_meta["cluster"].replace(wrong, correct)

            # correct the subclasses in the full meta data
            full_meta["subclass"] = full_meta["subclass"].replace("L4", "L4 IT")
            full_meta["subclass"] = full_meta["subclass"].replace("NP", "L5 NP")

            # correct the non-unique color for subclasses Sncg and Serpinf1, which both have color #8510C0
            # get mean color for Serpinf1
            def hex_to_rgb(hex_color):
                """
                Convert a hexadecimal color to an RGB tuple.
                """
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

            def rgb_to_hex(rgb_color):
                """
                Convert an RGB tuple to a hexadecimal color.
                """
                return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

            color1_serpinf1 = hex_to_rgb(cell_type_meta["cluster_color"].values[
                                             np.where(cell_type_meta["cluster_label"] == "Serpinf1 Clrn1")[0]][0])
            color2_serpinf1 = hex_to_rgb(cell_type_meta["cluster_color"].values[
                                             np.where(cell_type_meta["cluster_label"] == "Serpinf1 Aqp5 Vip")[0]][0])

            mean_color = tuple(np.mean([color1_serpinf1, color2_serpinf1], axis=0).astype(int))
            mean_color_hex = rgb_to_hex(mean_color)
            assert mean_color_hex not in cell_type_meta["subclass_color"].values

            mask = cell_type_meta["subclass_label"] == "Serpinf1"
            cell_type_meta.loc[mask, "subclass_color"] = [mean_color_hex] * mask.sum()

            ############################################################################
            # end of corrections
            ############################################################################

            # Create a dictionary to store the index of each element in the superset
            full_idx = {value: idx for idx, value in enumerate(full_meta["sample_name"].values)}

            # Create the mapping from subset index to superset index
            id_to_full_id = {idx: full_idx[value] for idx, value in enumerate(clusterInfo["sample_name"].values)}

            # Check that the mapping is correct by comparing the cluster names in full_meta and clusterInfo
            for i, name in enumerate(clusterInfo["sample_name"].values):
                full_id = id_to_full_id[i]
                assert name == full_meta["sample_name"].values[full_id]

            mapped_idx = [id_to_full_id[i] for i in range(len(clusterInfo["sample_name"].values))]

            # get interesting meta data and put them in the same shape
            cls = full_meta["class"].values[mapped_idx]
            subcls = full_meta["subclass"].values[mapped_idx]
            brain_region = full_meta["brain_region"].values[mapped_idx]
            brain_subregion = full_meta["brain_subregion"].values[mapped_idx]
            clusters_full = full_meta["cluster"].values[mapped_idx]

            # check that the mapping is correct
            assert np.all(clusters_full == clusterInfo["cluster_label"].values)


            class_labels = np.unique(cell_type_meta["class_label"].values)
            subclass_labels = np.unique(cell_type_meta["subclass_label"].values)

            class_colors = np.unique(cell_type_meta["class_color"].values)
            subclass_colors = np.unique(cell_type_meta["subclass_color"].values)

            class_label_to_id = {label: idx for idx, label in enumerate(class_labels)}
            class_id = np.array([class_label_to_id[label] for label in cls])

            subclass_label_to_id = {label: idx for idx, label in enumerate(subclass_labels)}
            subclass_id = np.array([subclass_label_to_id[label] for label in subcls])

            # superflous bc cluster info already read from the other meta file
            #cluster_labels = cell_type_meta["cluster_label"].values
            #cluster_colors = cell_type_meta[
            #    "cluster_color"].values
            #cluster_label_to_id = {label: idx for idx, label in enumerate(cluster_labels)}
            #cluster_id = np.array([cluster_label_to_id[label] for label in clusters_full])

            brain_region_labels = np.unique(brain_region)
            brain_region_colors = np.array(["r", "b"])
            brain_region_label_to_id = {label: idx for idx, label in enumerate(brain_region_labels)}
            brain_region_id = np.array([brain_region_label_to_id[label] for label in brain_region])

            brain_subregion_labels = np.unique(brain_subregion)
            brain_subregion_colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(brain_subregion_labels)))
            brain_subregion_label_to_id = {label: idx for idx, label in enumerate(brain_subregion_labels)}
            brain_subregion_id = np.array([brain_subregion_label_to_id[label] for label in brain_subregion])

            # put everything in the dictionary
            tasic2018["classNames"] = class_labels
            tasic2018["classColors"] = class_colors
            tasic2018["classes"] = class_id

            tasic2018["subclassNames"] = subclass_labels
            tasic2018["subclassColors"] = subclass_colors
            tasic2018["subclasses"] = subclass_id

            tasic2018["brainRegionNames"] = brain_region_labels
            tasic2018["brainRegionColors"] = brain_region_colors
            tasic2018["brainRegions"] = brain_region_id

            tasic2018["brainSubregionNames"] = brain_subregion_labels
            tasic2018["brainSubregionColors"] = brain_subregion_colors
            tasic2018["brainSubregions"] = brain_subregion_id
        except FileNotFoundError:
            print("Detailled meta data not found. Download from https://www.nature.com/articles/s41586-018-0654-5#Sec34")

        save_dict(tasic2018, os.path.join(data_dir, "tasic2018.pkl"))

        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

        importantGenesTasic2018 = geneSelection(
            tasic2018['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018['genes'], plot=False)

        librarySizes = np.sum(tasic2018['counts'], axis=1)

        #X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
        X = (tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6).log1p() # / np.log(2)  # is supposed to be log_2 (1+x)
        X = X.toarray()  # bc np.array() does not work with sparse matrices
        X = X - X.mean(axis=0)
        U, s, V = np.linalg.svd(X, full_matrices=False)
        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        X = X[:, np.argsort(s)[::-1]][:, :50]

        x = X
        y = tasic2018["clusters"]
        d = tasic2018

        np.save(os.path.join(data_dir, "pca50.npy"), x)
        np.save(os.path.join(data_dir, "labels.npy"), y)
        print("...done.")
    return x, y, d


def load_tasic3000(root_dataset):
    data_dir = os.path.join(root_dataset, "tasic")

    try:
        x = np.load(os.path.join(data_dir, "log_imp_genes.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "tasic2018.pkl"))
    except FileNotFoundError:
        urls = ["http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985",
                "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179",
                "https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/raw/allen/tasic2018/sample_heatmap_plot_data.csv"
                ]

        file_names = ["mouse_VISp_gene_expression_matrices_2018-06-14.zip",
                      "mouse_ALM_gene_expression_matrices_2018-06-14.zip",
                      "sample_heatmap_plot_data.csv"
                      ]

        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)

            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, file_name.split("/")[-1].strip(".zip")))

        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        file_name_VISp = os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv")
        counts1, genes1, cells1 = sparseload(file_name_VISp)

        file_name_ALM = os.path.join(data_dir, "mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv")
        counts2, genes2, cells2 = sparseload(file_name_ALM)

        counts = scipy.sparse.vstack((counts1, counts2), format='csc')

        cells = np.concatenate((cells1, cells2))

        if np.all(genes1==genes2):
            genes = np.copy(genes1)

        genesDF = pd.read_csv(os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14",
                                           "mouse_VISp_2018-06-14_genes-rows.csv"))
        ids = genesDF['gene_entrez_id'].tolist()
        symbols = genesDF['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv(os.path.join(data_dir, "sample_heatmap_plot_data.csv"))
        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)

        ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
        counts = counts[ind, :]

        areas = (ind < cells1.size).astype(int)

        clusters = clusters - 1

        tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                     'clusterColors': clusterColors, 'clusterNames': clusterNames}

        save_dict(tasic2018, os.path.join(data_dir, "tasic2018.pkl"))

        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

        importantGenesTasic2018 = geneSelection(
            tasic2018['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018['genes'], plot=False)

        librarySizes = np.sum(tasic2018['counts'], axis=1)
        X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
        X = np.array(X)

        y = tasic2018["clusters"]
        d = tasic2018

        np.save(os.path.join(data_dir, "log_imp_genes.npy"), X)
        np.save(os.path.join(data_dir, "labels.npy"), y)

    return x, y, d


def load_tasic_orange(root_path):
    x, y, d = load_tasic(root_path)
    mask_Pvalb = np.char.startswith(d["clusterNames"][d['clusters']], "Pvalb")
    mask_Sst = np.char.startswith(d["clusterNames"][d['clusters']], "Sst")

    mask = mask_Pvalb | mask_Sst

    d["clusters"] = d["clusters"][mask]
    d["counts"] = d["counts"][mask]
    d["areas"] = d["areas"][mask]

    return x[mask], y[mask], d

def load_tasic_purple(root_path):
    x, y, d = load_tasic(root_path)

    mask_Vip = np.char.startswith(d["clusterNames"][d['clusters']], "Vip")
    mask_Sncg = np.char.startswith(d["clusterNames"][d['clusters']], "Sncg")
    mask_Serpinf1 = np.char.startswith(d["clusterNames"][d['clusters']], "Serpinf1")
    
    mask = mask_Vip | mask_Sncg | mask_Serpinf1

    d["clusters"] = d["clusters"][mask]
    d["counts"] = d["counts"][mask]
    d["areas"] = d["areas"][mask]

    return x[mask], y[mask], d

def load_tasic_purple_lamp5(root_path):
    x, y, d = load_tasic(root_path)

    mask_Vip = np.char.startswith(d["clusterNames"][d['clusters']], "Vip")
    mask_Sncg = np.char.startswith(d["clusterNames"][d['clusters']], "Sncg")
    mask_Serpinf1 = np.char.startswith(d["clusterNames"][d['clusters']], "Serpinf1")
    mask_Lamp5 = np.char.startswith(d["clusterNames"][d['clusters']], "Lamp5")

    mask_Lamp5_Lhx6 = np.char.startswith(d["clusterNames"][d['clusters']], "Lamp5 Lhx6")  
    mask_lamp5_wo_Lhx6 = [a and not b for a, b in zip(mask_Lamp5, mask_Lamp5_Lhx6)]
    mask = mask_Vip | mask_Sncg | mask_Serpinf1 | mask_lamp5_wo_Lhx6
    
    d["clusters"] = d["clusters"][mask]
    d["counts"] = d["counts"][mask]
    d["areas"] = d["areas"][mask]

    return x[mask], y[mask], d


def load_small_tasic(root_path, dataset, seed=0, size=1000):
    if dataset == "tasic":
        x, y, d = load_tasic(root_path)
    elif dataset == "tasic_orange":
        x, y, d = load_tasic_orange(root_path)
    elif dataset == "tasic_purple":
        x, y, d = load_tasic_purple(root_path)
    elif dataset == "tasic3000":
        x, y, d = load_tasic3000(root_path)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    np.random.seed(seed)
    idx = np.random.choice(len(x), replace=False, size=size)

    d["clusters"] = d["clusters"][idx]
    d["counts"] = d["counts"][idx]
    d["areas"] = d["areas"][idx]
    d["idx"] = idx
    return x[idx], y[idx], d


def load_mca_ss2(root_path):
    data_dir = os.path.join(root_path, "mca_ss2")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    try:
        x = np.load(os.path.join(data_dir, "mca_ss2.data.npy"))
        y = np.load(os.path.join(data_dir, "mca_ss2.labels.npy"))
        d = load_dict(os.path.join(data_dir, "meta_mca_ss2.pkl"))
    except FileNotFoundError:
        print("Downloading MCA Smartseq2 data...", end="", flush=True)
        urls = ["https://github.com/vhowick/MalariaCellAtlas/raw/v1.0/Expression_Matrices/Smartseq2/SS2_tmmlogcounts.csv.zip",
                "https://raw.githubusercontent.com/vhowick/MalariaCellAtlas/v1.0/Expression_Matrices/Smartseq2/SS2_pheno.csv"]
        for url in urls:
            file_name =  os.path.join(data_dir, url.split("/")[-1])
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(data_dir)

        file_name_pp = os.path.join(data_dir, "SS2_tmmlogcounts.csv")
        x = pd.read_csv(file_name_pp, encoding='latin-1', header=0, index_col=0).to_numpy().T

        file_name_pheno = os.path.join(data_dir, "SS2_pheno.csv")
        clusters = pd.read_csv(file_name_pheno, encoding='latin-1')["ShortenedLifeStage2"].to_numpy()

        # numeric labels
        cluster_names  = np.unique(clusters)
        cluster_names_to_labels = {cluster_names[i]: i for i in range(len(cluster_names))}
        y = np.array([cluster_names_to_labels[clusters[i]] for i in range(len(clusters))])

        # dicts for plotting
        cluster_names_to_print_names = {
            "bbSpz" :"Injected sporozoite",
            "EEF":"Liver stage",
            "Merozoite":"Merizoite",
            "oocyst":"Oocyst",
            "ook" :"Bolus okinete",
            "ooSpz" :"????",
            "Ring":"Ring",
            "sgSpz":"Gland sporozoite",
            "Schizont":"Schizont",
            "Male":"Male gametocyte",
            "Female":"Female gametocyte",
            "ookoo" :"Okinete/oocyst",
            "Trophozoite":"Trophozoite"
        }

        cluster_names_to_colors = {
            "bbSpz" : "#000080",
            "EEF":"#ff8c00",
            "Merozoite":"#ffb6c1",
            "oocyst":"#4682b4",
            "ook" : "#00868b",
            "ooSpz" :"#87cefa",
            "Ring":"#ff69b4",
            "sgSpz":"#4169e1",
            "Schizont":"#d02090",
            "Male":"#a020f0",
            "Female":"#551a8b",
            "ookoo":"#48d1cc",
            "Trophozoite":"#ee82ee"

        }

        d = {"cluster_names": {i: cluster_names[i] for i in range(len(cluster_names))},
             "cluster_print_names": {i: cluster_names_to_print_names[cluster_names[i]] for i in range(len(cluster_names))},
             "cluster_colors": {i: cluster_names_to_colors[cluster_names[i]] for i in range(len(cluster_names))}}

        outputfile = "mca_ss2"
        np.save(os.path.join(data_dir, outputfile + ".data.npy"), x)
        np.save(os.path.join(data_dir, outputfile + ".labels.npy"), y)
        save_dict(d, os.path.join(data_dir, "meta_mca_ss2.pkl"))

        print("done")
    return x, y, d


def load_mca_ss2_idc(root_path):
    x, y, d = load_mca_ss2(root_path)

    # labels in IDC
    labels_idc = ["Merozoite", "Ring", "Trophozoite", "Schizont"]

    # filter data
    idx = np.array([d["cluster_names"][cluster_id] in labels_idc for cluster_id in y])

    x = x[idx]
    y = y[idx]
    for i in np.unique(y):
        if d["cluster_names"][i] not in labels_idc:
            del d["cluster_names"][i]
            del d["cluster_print_names"][i]
            del d["cluster_colors"][i]
    return x, y, d


dataset2url_cc = {"neurosphere": "https://zenodo.org/record/5519841/files/neurosphere.qs",
               "hippocampus": "https://zenodo.org/record/5519841/files/hipp.qs",
               "HeLa2": "https://zenodo.org/record/5519841/files/HeLa2.qs",
               "pancreas": "https://zenodo.org/record/5519841/files/endo.qs",
               "pallium": "https://storage.googleapis.com/linnarsson-lab-tmp/Cortex_EMX1_louvain3_passedQC_PostM_rev1.h5ad"}


def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully from {url}.")
    else:
        print(f"Failed to download file from {url}. Status code: {response.status_code}")


def download_cc_file(dataset, root_path):
    data_dir = os.path.join(root_path, dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download_file(dataset2url_cc[dataset], os.path.join(data_dir, f"{dataset}.qs"))


def load_cc_dataset(root_path, dataset, representation="tricycleEmbedding"):
    data_dir = os.path.join(root_path, dataset)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    try:
        with h5py.File(os.path.join(data_dir, dataset+".h5"), "r") as f:
            assert representation in f.keys(), f"representation {representation} not found, only {f.keys()} available"
            x = f[representation][:].T
            stages = np.array([s.decode("utf-8") for s in f["CCStage"][:]])
            if "cell_type" in f.keys():
                cell_types = np.array([s.decode("utf-8") for s in f["cell_type"][:]])
                cell_types_exist = True
            else:
                cell_types_exist = False
            theta = f["tricyclePosition"][:]
    except:
        # download dataset
        qs_file = os.path.join(root_path, dataset, f"{dataset}.qs")
        if not os.path.exists(qs_file):
            download_cc_file(dataset=dataset, root_path=root_path)

        # pancreas dataset does not have the tricycle and GOPCA embedding precomputed, so we need to compute it ourselves
        if dataset == "pancreas":
            # clone required git repos
            tricycle_path = f"{os.path.dirname(os.path.realpath(__file__))}/tricycle"
            if not os.path.exists(tricycle_path):
                subprocess.run(["git", "clone", "https://github.com/hansenlab/tricycle.git", tricycle_path])
            tricycle_fig_path = f"{os.path.dirname(os.path.realpath(__file__))}/tricycle_paper_figs"
            if not os.path.exists(tricycle_fig_path):
                subprocess.run(["git", "clone", "https://github.com/hansenlab/tricycle_paper_figs.git", tricycle_fig_path])

            # compute the GOPCA and tricycle embedding representations
            subprocess.run(["conda", "run", "-n", "oneR",
                            "Rscript", f"{os.path.dirname(os.path.realpath(__file__))}/pancreas.R", str(root_path), str(os.path.dirname(os.path.realpath(__file__)))])

        # copy data from .qs to .h5
        # note that this does not work if this function is called from a jupyter notebook or a tmux environment
        subprocess.run(["conda", "run", "-n", "oneR",
                        "Rscript", f"{os.path.dirname(os.path.realpath(__file__))}/cc_dataset.R", str(root_path), dataset])

        # now that the data is available in .h5, load it
        with h5py.File(os.path.join(data_dir, dataset+".h5"), "r") as f:
            assert representation in f.keys(), f"representation {representation} not found, only {f.keys()} available"
            x = f[representation][:].T
            stages = np.array([s.decode("utf-8") for s in f["CCStage"][:]])
            if "cell_type" in f.keys():
                cell_types = np.array([s.decode("utf-8") for s in f["cell_type"][:]])
                cell_types_exist = True
            else:
                cell_types_exist = False
            theta = f["tricyclePosition"][:]
        #return load_cc_dataset(root_path=root_path, dataset=dataset, representation=representation)

    cc_colors = {"G1.S": '#B2627C', "S": '#F29360', "G2": '#FCEA64', "G2.M": '#86BBD8', "M.G1": '#8159ba', "NA": "gray"}

    unique_stages = np.unique(stages)

    stage_to_y = {unique_stages[i]: i for i in range(len(unique_stages))}
    y = np.array([stage_to_y[stage] for stage in stages])
    y_to_stage = {v: k for k, v in stage_to_y.items()}

    reordered_stages = np.array([y_to_stage[i] for i in np.unique(y)])
    reordered_colors = np.array([cc_colors[y_to_stage[i]] for i in np.unique(y)])

    d = {"stage_names": reordered_stages,
         "colors": reordered_colors,
         "theta": theta,}
    if cell_types_exist:
        d["cell_types"] = cell_types

    return x, y, d


def load_small_cc_dataset(root_path, dataset, representation="tri", seed=0):
    x, y, d = load_cc_dataset(root_path, dataset, representation)
    np.random.seed(seed)
    idx = np.random.choice(len(x), replace=False, size=1000)
    d["theta"] = d["theta"][idx]
    d["cell_types"] = d["cell_types"][idx]
    return x[idx], y[idx], d


def load_pallium_scVI(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI", "pallium_scVI.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        adata = sc.read(os.path.join(root_path, "pallium", "pallium.h5ad"))

        if not os.path.exists(os.path.join(root_path, "pallium_scVI")):
            os.mkdir(os.path.join(root_path, "pallium_scVI"))
        with h5py.File(os.path.join(root_path, "pallium_scVI", "pallium_scVI.h5"), "w") as file:
            file.create_dataset("scVI", data=np.array(adata.obsm["X_scVI"]))
            file.create_dataset("Subset", data=np.array(adata.obs["Subset"]))
            file.create_dataset("CellClass", data=np.array(adata.obs["CellClass"]))
            file.create_dataset("CellCycle", data=np.array(adata.obs["CellCycle"]))
            file.create_dataset("Cycling", data=np.array(adata.obs["Cycling"]))
            file.create_dataset("UMAP", data=np.array(adata.obsm["X_umap"]))
        x, y, d = load_pallium_scVI(root_path)

    return x, y, d


def load_pallium_scVI_10pcw(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_10pcw", "pallium_scVI_10pcw.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        x, y, d = load_pallium_scVI(root_path)

        mask = d["Subset"] == b"10wk"
        new_dataset = "pallium_scVI_10pcw"

        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])

        # reload, now the try block should work out
        x, y, d = load_pallium_scVI_10pcw(root_path)
    return x, y, d


def load_pallium_scVI_10pcw_cycling(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_10pcw_cycling", "pallium_scVI_10pcw_cycling.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:

        x, y, d = load_pallium_scVI_10pcw(root_path)

        mask = d["Cycling"]

        new_dataset = "pallium_scVI_10pcw_cycling"
        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])

        # reload, now the try block should work out
        x, y, d = load_pallium_scVI_10pcw_cycling(root_path)

    return x, y, d


def load_pallium_scVI_IPC(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_IPC", "pallium_scVI_IPC.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        x, y, d = load_pallium_scVI(root_path)

        mask = y == b'Neuronal IPC'
        new_dataset = "pallium_scVI_IPC"
        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])
        x, y, d = load_pallium_scVI_IPC(root_path)

    return x, y, d



def load_subsampled_pallium(root_path, dataset, seed=0, n=1000):
    x, y, _, _, d = load_dataset(root_path, dataset)
    np.random.seed(seed)
    mask = np.random.choice(x.shape[0], n, replace=False)
    return x[mask], y[mask], {key: value[mask] for key, value in d.items()}


def load_subsampled_10x(root_path, dataset, seed=0, n=1000):
    x, y, _, _, d = load_dataset(root_path, dataset)
    
    np.random.seed(seed)
    mask = np.random.choice(x.shape[0], n, replace=False)
    
    sampled_d = d
    
    sampled_d["counts"] = sampled_d["counts"][mask]
    sampled_d["clusters"] = sampled_d["clusters"][mask]
    sampled_d["cells"] = sampled_d["cells"][mask]
    # newdataset = dataset+f"_sample_{seed}"
    # np.save(os.path.join(root_path, newdataset, "mask.npy"), mask)
    
    return x[mask], y[mask], sampled_d


def load_yao_10x_split(root_path, dataset, seed=0):
    data_dir = os.path.join(root_path, dataset)
    supdataset, i = dataset.split("_split_")
    i = int(i)
    try:
        x = np.load(os.path.join(data_dir, "pca50.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "yao.pkl"))
    except FileNotFoundError:
        if supdataset == "yao_10x_female_orange":
            x, y, d = load_yao_10x_female_orange(root_path)
            n_splits = 5
        elif supdataset =="yao_10x_female_purple":
            x, y, d = load_yao_10x_female_purple(root_path)
            n_splits = 5
        elif supdataset == "yao_10x_male_orange":
            x, y, d = load_yao_10x_male_orange(root_path)
            n_splits = 10
        elif supdataset == "yao_10x_male_purple":
            x, y, d = load_yao_10x_male_purple(root_path)
            n_splits = 10
        inx = np.arange(x.shape[0])

        shuffled_idx = np.copy(inx)
        
        np.random.seed(seed)
        np.random.shuffle(shuffled_idx)
        
        shuffled_idx = np.array_split(shuffled_idx, n_splits)
        x = x[shuffled_idx[i]]
        y = y[shuffled_idx[i]]
        
        # d["clusters"] = [d["clusters"][a] for a in shuffled_idx[i]]
        # d["counts"] = [d["counts"][a] for a in shuffled_idx[i]]
        # d["cells"] = [d["cells"][a] for a in shuffled_idx[i]]
        d["clusters"] = d["clusters"][shuffled_idx[i]] #.toarray()
        d["counts"] = d["counts"][shuffled_idx[i]]#.toarray()
        d["cells"] = d["cells"][shuffled_idx[i]]#.toarray()
        d['shuffled_idx'] = shuffled_idx[i]#.toarray()
        
        save_dict(d, os.path.join(data_dir, "yao.pkl"))
        np.save(os.path.join(data_dir, "pca50.npy"), x)
        np.save(os.path.join(data_dir, "labels.npy"), y)
    return x, y, d

def load_k49(root_dataset):
    data_dir = os.path.join(root_dataset, "k49")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(os.path.join(data_dir, "k49-train-imgs.npz")):
        urls = ["http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz"]
        for url in urls:
            name = url.split("/")[-1]
            file_name = os.path.join(data_dir, name)
            urllib.request.urlretrieve(url, file_name)

    with np.load(os.path.join(data_dir, "k49-train-imgs.npz")) as data:
        x_train = data["arr_0"]
    with np.load(os.path.join(data_dir, "k49-test-imgs.npz")) as data:
        x_test = data["arr_0"]
    x = np.concatenate([x_train, x_test])
    x = x.reshape(len(x), -1)

    with np.load(os.path.join(data_dir, "k49-train-labels.npz")) as data:
        y_train = data["arr_0"]

    with np.load(os.path.join(data_dir, "k49-test-labels.npz")) as data:
        y_test = data["arr_0"]
    y = np.concatenate([y_train, y_test])

    return x, y


def load_toy(root_path):
    data_dir = os.path.join(root_path, "toy")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([0.5, np.sqrt(3)/2])

    x= np.stack([p1, p2, p3])
    y = np.arange(3, dtype=int)

    return x, y


# complete loader:
def load_dataset(root_path, dataset, k=15, seed=None):
    # load dataset
    if not os.path.exists(os.path.join(root_path, dataset)):
        os.mkdir(os.path.join(root_path, dataset))
    if dataset == "pendigits":
        x, y = load_pendigits(root_path)

    elif dataset == "mnist":
        x, y = load_mnist(root_path)

    elif dataset.startswith("mnist_"):
        l = dataset.split("_")
        ind_seed = l.index("seed")
        seed = int(l[ind_seed + 1])
        ind_size = l.index("size")
        size = int(l[ind_size + 1])
        x, y = load_small_mnist(root_path, seed, size)

    elif dataset.startswith("imba_mnist"):
        l = dataset.split("_")
        ind_seed = l.index("seed")
        seed = int(l[ind_seed + 1])
        mode = l[2]
        if mode == "lin":
            x, y = load_imba_mnist(root_path,
                                   props = 1.0 - np.arange(10) / 10,
                                   seed=seed)
        elif mode == "odd":
            x, y = load_imba_mnist(root_path,
                                   props = 5*[1.0, 0.1],
                                   seed = seed)
        else:
            raise NotImplementedError(f"Subsampling mode {mode} is not implemented.")
    elif dataset == "cifar10":
        x, y = load_cifar10(root_path)
    elif dataset == "human-409b2":
        x, y, d = load_human(root_path)
    elif dataset == "zebrafish":
        x, y = load_zebrafish(root_path)
    elif dataset == "c_elegans":
        x, y = load_c_elegans(root_path)
    elif dataset == "k49":
        x, y = load_k49(root_path)
    elif dataset == "toy":
        x, y = load_toy(root_path)
    elif dataset == "tasic":
        x, y, d = load_tasic(root_path)
    elif dataset == "yao":
        x, y, d = load_yao(root_path) # not used anymore, since we wanna seperate 10x and smart seq in yao
    elif dataset == "yao_smart":
        x, y, d = load_yao_smart(root_path)
    elif dataset == "yao_smart_purple":
        x, y, d = load_yao_smart_purple(root_path)
    elif dataset == "yao_smart_orange":
        x, y, d = load_yao_smart_orange(root_path)
    elif dataset == "yao_10x":
        x, y, d = load_yao_10x(root_path)
        
    elif dataset == "yao_10x_female":
        x, y, d = load_yao_10x_female(root_path)
    elif dataset == "yao_10x_female_orange":
        x, y, d = load_yao_10x_female_orange(root_path)
    elif dataset == "yao_10x_female_purple":
        x, y, d = load_yao_10x_female_purple(root_path)
    elif dataset == "yao_10x_female_orange_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_female_orange", seed=seed, n=10000)
    elif dataset == "yao_10x_female_purple_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_female_purple", seed=seed, n=10000)
    elif dataset == "yao_10x_female_orange_5k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_female_orange", seed=seed, n=5000)
    elif dataset == "yao_10x_female_purple_5k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_female_purple", seed=seed, n=5000)
        
    elif dataset == "yao_10x_male":
        x, y, d = load_yao_10x_male(root_path)
    elif dataset == "yao_10x_male_orange": 
        x, y, d = load_yao_10x_male_orange(root_path)
    elif dataset == "yao_10x_male_purple":
        x, y, d = load_yao_10x_male_purple(root_path)
        
    elif dataset == "yao_10x_male_green_56IT":
        x, y, d = load_yao_10x_male_green_56IT(root_path)
    elif dataset == "yao_10x_male_green_56IT_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_green_56IT", seed=0, n=10000)
    elif dataset == "yao_10x_male_green_56ITCTX":
        x, y, d = load_yao_10x_male_green_56ITCTX(root_path)
    elif dataset == "yao_10x_male_green_56ITCTX_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_green_56ITCTX", seed=0, n=10000)
    elif dataset == "yao_10x_male_green_56IT2":
        x, y, d = load_yao_10x_male_green_56IT2(root_path)
    elif dataset == "yao_10x_male_green_56IT2_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_green_56IT2", seed=0, n=10000)
    
    elif dataset == "yao_10x_male_purple_20k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_purple", seed=seed, n=20000)
    elif dataset == "yao_10x_male_orange_20k": 
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_orange", seed=seed, n=20000)
        
    elif dataset == "yao_10x_male_purple_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_purple", seed=seed, n=10000)
    elif dataset == f"yao_10x_male_orange_10k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_orange", seed=seed, n=10000)
    
    elif dataset == "yao_10x_male_purple_0_10k":  
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_purple", seed=0, n=10000)
    elif dataset == "yao_10x_male_orange_0_10k": 
        # to name differently from yao_10x_male_orange_10k bc in the 2nd one sometimes resample from whole dataset, while in this one
        # bootstrap is from the 10k that already sampled...
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_orange", seed=0, n=10000)

    elif dataset == "yao_10x_male_purple_5k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_purple", seed=seed, n=5000)
    elif dataset == "yao_10x_male_orange_5k":
        x, y, d = load_subsampled_10x(root_path, "yao_10x_male_orange", seed=seed, n=5000)
    elif "_split_" in dataset: #= "yao_10x_{male/female}_{orange/purple}_split_{0-4 if female, 0-9 if male}"
        x, y, d = load_yao_10x_split(root_path, dataset = dataset, seed=0)
        
    elif dataset == "tasic3000":
        x, y, d = load_tasic3000(root_path)
    elif dataset == "tasic_orange":
        x, y, d = load_tasic_orange(root_path)
    elif dataset == "tasic_purple": #this one is totally useless since it does not include lamp5....
        x, y, d = load_tasic_purple(root_path)
    elif dataset == "tasic_purple_lamp5":
        x, y, d = load_tasic_purple_lamp5(root_path)
    elif dataset == "tasic_small":
        x, y, d = load_small_tasic(root_path, "tasic", seed=seed, size=1000)
    elif dataset == "tasic_orange_small":
        x, y, d = load_small_tasic(root_path, "tasic_orange", seed=seed, size=1000)
    elif dataset == "tasic_purple_small":
        x, y, d = load_small_tasic(root_path, "tasic_purple", seed=seed, size=1000)
    elif dataset == "tasic3000_small":
        x, y, d = load_small_tasic(root_path, "tasic3000", seed=seed, size=1000)
    elif dataset == "mca_ss2":
        x, y, d = load_mca_ss2(root_path)
    elif dataset == "mca_ss2_idc":
        x, y, d = load_mca_ss2_idc(root_path)
    elif dataset == "neurosphere_pca":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="PCA30D")
    elif dataset == "neurosphere_gopca":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="GOPCA20D")
    elif dataset == "neurosphere_tricycle":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="tricycleEmbedding")
    elif dataset == "neurosphere_pca_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="PCA30D", seed=seed)
    elif dataset == "neurosphere_gopca_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="GOPCA20D", seed=seed)
    elif dataset == "neurosphere_tricycle_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="tricycleEmbedding", seed=seed)
    elif dataset == "hippocampus_pca":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="PCA30D")
    elif dataset == "hippocampus_gopca":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="GOPCA20D")
    elif dataset == "hippocampus_tricycle":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="tricycleEmbedding")
    elif dataset == "hippocampus_pca_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="PCA30D", seed=seed)
    elif dataset == "hippocampus_gopca_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="GOPCA20D", seed=seed)
    elif dataset == "hippocampus_tricycle_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="tricycleEmbedding", seed=seed)
    elif dataset == "HeLa2_pca":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="PCA30D")
    elif dataset == "HeLa2_gopca":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="GOPCA20D")
    elif dataset == "HeLa2_tricycle":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="tricycleEmbedding")
    elif dataset == "pancreas_pca":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="PCA30D")
    elif dataset == "pancreas_gopca":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="GOPCA20D")
    elif dataset == "pancreas_tricycle":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="tricycleEmbedding")
    elif dataset == "pallium_scVI":
        x, y, d = load_pallium_scVI(root_path)
    elif dataset == "pallium_scVI_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI", seed=seed, n=1000)
    elif dataset == "pallium_scVI_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI", seed=seed, n=5000)
    elif dataset == "pallium_scVI_10pcw":
        x, y, d = load_pallium_scVI_10pcw(root_path)
    elif dataset == "pallium_scVI_10pcw_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw", seed=seed, n=1000)
    elif dataset == "pallium_scVI_10pcw_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw", seed=seed, n=5000)
    elif dataset == "pallium_scVI_10pcw_cycling":
        x, y, d = load_pallium_scVI_10pcw_cycling(root_path)
    elif dataset == "pallium_scVI_10pcw_cycling_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw_cycling", seed=seed, n=1000)
    elif dataset == "pallium_scVI_10pcw_cycling_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw_cycling", seed=seed, n=5000)
    elif dataset == "pallium_scVI_IPC":
        x, y, d = load_pallium_scVI_IPC(root_path)
    elif dataset == "pallium_scVI_IPC_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_IPC", seed=seed, n=1000)
    elif dataset == "pallium_scVI_IPC_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_IPC", seed=seed, n=5000)
    else:
        raise NotImplementedError


    # get pca
    # load / compute and save 2D PCA for initialisation
    try:
        pca2 = np.load(os.path.join(root_path, dataset, "pca2.npy"))
    except FileNotFoundError:
        pca_projector = PCA(n_components=2)
        pca2 = pca_projector.fit_transform(np.array(x))
        np.save(os.path.join(root_path, dataset, "pca2.npy"), pca2)

    # get skkn graph
    knn_file_name = os.path.join(root_path,
                                 dataset,
                                 f"sknn_graph_k_{k}_metric_euclidean.npz")

    try:
        sknn_graph = scipy.sparse.load_npz(knn_file_name)
    except IOError:
        x_for_knn = x
        if dataset.startswith("mnist") or dataset == "k49" or dataset == "cifar10":
            try:
                pca50 = np.load(os.path.join(root_path, dataset, "pca50.npy"))
            except FileNotFoundError:
                pca_50_projector = PCA(n_components=50, random_state=0)
                pca50 = pca_50_projector.fit_transform(x)
                np.save(os.path.join(root_path, dataset, "pca50.npy"), pca50)
            x_for_knn = pca50

        knn_graph = kNN_graph(x_for_knn.astype("float"),
                              k,
                              metric="euclidean").cpu().numpy().flatten()
        knn_graph = scipy.sparse.coo_matrix((np.ones(len(x) * k),
                                             (np.repeat(np.arange(x.shape[0]), k),
                                              knn_graph)),
                                            shape=(len(x), len(x)))
        sknn_graph = knn_graph.maximum(knn_graph.transpose()).tocoo()
        scipy.sparse.save_npz(knn_file_name, sknn_graph)

    if 'd' in locals():
        return x, y, sknn_graph, pca2, d
    else:
        return x, y, sknn_graph, pca2