# ABC-MC  
Public code for the paper *“Approximate Bayesian Computation with Statistical Distances for Model Selection”* (arXiv:2410.21603) 

##  Repository structure  

├── abc_utils.py # Core functions
├── boxplots.py # Utility to generate comparative boxplots of probabilities
├── confusionmatrices.py # Script for confusion-matrix construction from method outputs
├── normal_example/
│ ├── … # Scripts and data for the “normal” example
├── expofamily_example/
│ ├── … # Scripts + data for the exponential‐family example
├── g-and-k_example/
│ ├── … # Scripts + data for the g-and-k distribution example
├── toad_example/
│ ├── … # Scripts + data for the toad movement example
└── README.md # This file


## Purpose  
This repository provides code and examples for applying ABC (Approximate Bayesian Computation) using statistical distances (CvM, MMD, Wasserstein) for model selection across different illustrative examples:
- A normal-distribution example 
- An exponential-family example 
- A g-and-k quantile-distribution example 
- A toad movement simulation example 

## Requirements  
- Python 3 (≥3.7 recommended) 
- Key Python packages: `numpy`, `scipy`, `joblib`, `multiprocessing` 
- R (for the semi‐automatic ABC scripts) 
- R packages: `abctools`, `reticulate` (for loading `.npz` when needed) 
- Git for version control 

##  Example workflow  
### 1. Generate observed datasets  
For example, using `normal_example/normal_example_distanceABC.py`: generate ~100 observed datasets and save them (e.g., `data/observed_datasets.npz`). 

### 2. Simulate training/ABC data  
Simulate datasets under competing models, compute summary statistics, apply `run_abc_for_one_observed()` to each observed dataset, collect posterior model probabilities and parameter estimates. 

### 3. Save results  
Example script will output CSV files for:
- Model probabilities (e.g., `*_probabilities.csv`) 
- Parameter estimates (e.g., `*_params.csv`) 

### 4. Visualise / Compare  
Use `boxplots.py` and `confusionmatrices.py` to compare methods (e.g., ABC-CvM, ABC-MMD, ABC-Wass, NN, ABC-SA, ABC-QDA). 

## References  
- [arXiv:2410.21603](https://arxiv.org/abs/2410.21603) — The main paper describing the methodology 



