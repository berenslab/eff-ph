# Persistent homology for high-dimensional data based on spectral methods
Repository accompanying the paper 

**Persistent homology for high-dimensional data based on spectral methods** NeurIPS 2024 ([openreview](https://openreview.net/forum?id=ARV1gJSOzV))  
Sebastian Damrich, Philipp Berens, Dmitry Kobak

```
@article{damrich2024persistent,
  title={Persistent homology for high-dimensional data based on spectral methods},
  author={Damrich, Sebastian and Berens, Philipp and Kobak, Dmitry},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2024}
}
``` 

<p align="center"> <img alt="PH with Effective resistance vs Euclidean distance on Circle" src="/figures/fig_1.png">

## Usage
Compute the persistent homology of a toy dataset with `compute_ph.py`, of toy datasets with outliers with `compute_ph_outliers.py`
and that of a single-cell dataset with `compute_ph_real_data.py`. For the cycle matching experiments run the script 
`compute_matchings.py` Changing the dataset in the top of the script allows to compute the persistent homology of different datasets.
```
cd scripts
python compute_ph.py
```

Create the figures of the paper with the various `fig_*.ipynb` notebooks. The notebooks create the following figures:
- Figure 1: `fig_1.ipynb`
- Figure 2: `fig_ph.ipynb`
- Figure 3: `fig_vary_dim_mds.ipynb`
- Figure 4: `fig_spectral_intuition.ipynb`
- Figure 5: `fig_spectral.ipynb`
- Figure 6: `fig_circle.ipynb`
- Figure 7: `fig_datasets.ipynb`
- Figure 8: `fig_dims.ipynb`
- Figure 9, 10: `fig_real_data.ipynb`
- Figure S1: `fig_dims.ipynb`
- Figure S2: `fig_pca.ipynb`
- Figure S3: `fig_wide_gap.ipynb`
- Figure S4, S5: `fig_cycle_matching.ipynb`
- Figure S7, S8: `fig_spectral.ipynb`
- Figure S9: `fig_real_data.ipynb`
- Figure S10, S11: `fig_circle.ipynb`
- Figure S12, S13: `fig_toy_datasets.ipynb`
- Figure S14: `fig_sc_datasets.ipynb`
- Figure S15: `fig_sensitivity.ipynb`
- Figure S16: `fig_outliers.ipynb`
- Figure S17, S18: `fig_high_dim_UMAP.ipynb`
- Figure S19: `fig_real_data.ipynb`
- Figure S20: `fig_circle.ipynb`
- Figure S21: `fig_datasets.ipynb`
- Figure S22: `fig_circle.ipynb`
- Figures S23-S30, S33: `fig_all_methods_on_toy.ipynb`
- Figure S31, S32: `fig_torus_high_n.ipynb`
- Figure S29: `fig_real_data.ipynb`


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

Clone the repository `interval-matching` for the cycle matching experiments and compile the two C++ files:
```
cd ..
git clone https://github.com/inesgare/interval-matching.git
cd modified ripser/ripser-image-persistence-simple
make 
cd ../ripser-tight-representative-cycles
make
cd ../..
```

Clone the repository `vis_utils`
```
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


