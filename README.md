# Persistent homology for high-dimensional data based on spectral methods
Repository accompanying the paper [Persistent homology for high-dimensional data based on spectral methods](https://arxiv.org/abs/2311.03087)

<p align="center"> <img alt="PH with Effective resistance vs Euclidean distance on Circle" src="/figures/fig_1.png">

## Usage
Compute the persistent homology of a toy dataset with `compute_ph.py`, of toy datasets with outliers with `compute_ph_outliers.py`
and that of a single-cell dataset with `compute_ph_real_data.py`. Changing the dataset in the top of the script allows 
to compute the persistent homology of different datasets.
```
cd scripts
python compute_ph.py
```

Create the figures of the paper with the various `fig_*.ipynb` notebooks. The notebooks create the following figures:
- Figure 1: `fig_1.ipynb`
- Figure 2: `fig_ph.ipynb`
- Figure 3: `fig_vary_dim_mds.ipynb`
- Figure 4: `fig_circle.ipynb`
- Figure 5: `fig_datasets.ipynb`
- Figure 6: `fig_dims.ipynb`
- Figure 7: `fig_real_data.ipynb`
- Figure 8: `fig_real_data.ipynb`
- Figure 9: `fig_spectral.ipynb`
- Figure S1: `fig_circle.ipynb`
- Figure S2: `fig_circle.ipynb`
- Figure S3: `fig_toy_datasets.ipynb`
- Figure S4: `fig_toy_datasets.ipynb`
- Figure S5: `fig_sc_datasets.ipynb`
- Figure S6: `fig_outliers.ipynb`
- Figure S7: `fig_high_dim_UMAP.ipynb`
- Figure S8: `fig_high_dim_UMAP.ipynb`
- Figure S9: `fig_real_data.ipynb`
- Figure S10: `fig_circle.ipynb`
- Figure S11: `fig_datasets.ipynb`
- Figure S12: `fig_circle.ipynb`
- Figures S13-S20, S22: `fig_all_methods_on_toy.ipynb`
- Figure S21: `fig_torus_high_n.ipynb`
- Figure S21: `fig_torus_high_n.ipynb`
- Figure S23: `fig_real_data.ipynb`
- Figure S24: `fig_Lp.ipynb`


## Installation
Clone the repository
```
git clone https://github.com/berenslab/eff-ph.git
```

Create a conda python environment
```
cd eff-ph
conda env create -f environment.yml
```

Install the utils:
```
cd ../eff-ph
python setup.py install
```

Clone the repository `ripser` and compile it:
```
cd ..
git clone -b representative-cycles https://github.com/Ripser/ripser.git
cd risper
make
``` 


Clone the repository `vis_utils`
```
cd ..
git clone https://github.com/sdamrich/vis_utils.git --branch eff-ph-arxiv-v1 --single-branch
```

Create the conda R environment (for loading some single-cell datasets)
```
cd vis_utils
conda create -f r_env.yml
```

Install `vis_utils`
```
conda activate eff-ph
python setup.py install
```


