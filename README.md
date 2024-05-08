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
- Figure 4: `fig_spectral.ipynb`
- Figure 5: `fig_circle.ipynb`
- Figure 6: `fig_datasets.ipynb`
- Figure 7: `fig_dims.ipynb`
- Figure 8, 9: `fig_real_data.ipynb`
- Figure S1: `fig_dims.ipynb`
- Figure S3, S4: `fig_spectral.ipynb`
- Figure S5: `fig_real_data.ipynb`
- Figure S6, S7: `spectral.ipynb`
- Figure S8, S9: `fig_toy_datasets.ipynb`
- Figure S10: `fig_sc_datasets.ipynb`
- Figure S11: `fig_sensitivity.ipynb`
- Figure S12: `fig_outliers.ipynb`
- Figure S13, S14: `fig_high_dim_UMAP.ipynb`
- Figure S15: `fig_real_data.ipynb`
- Figure S16: `fig_circle.ipynb`
- Figure S17: `fig_datasets.ipynb`
- Figure S18: `fig_circle.ipynb`
- Figures S19-S26, S28: `fig_all_methods_on_toy.ipynb`
- Figure S27: `fig_torus_high_n.ipynb`
- Figure S29: `fig_real_data.ipynb`
- Figure S30: `fig_Lp.ipynb`


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


