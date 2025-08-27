Approximate Bayesian Computation with Statistical Distances for Model Selection

This repository contains Jupyter notebooks for reproducing the simulation study of the paper. 
The project generate posterior distributions of model probabilities and model parameters using full data ABC with various distance metrics 
(Cramér–von Mises, MMD, Wasserstein) and compare with ABC-Stat where the summary statistics are problem-specific. 

Repository Structure

1. analysis.ipynb
Extract results from the .npz files with the saved results for the posterior estimates of the model probabilities and the model parameters. Save .csv files and produce boxplots of model probabilities.

2. expo_family_example_exp.ipynb
File to reproduce the simulation for the exponential family model choice example. The file includes simulations from the exponential distribution as example, but can be modified for simulations from the log-normal or gamma distribution.

3. normal_example_m0.ipynb
File to reproduce the simulation for the normal hypothesis testing example. The file includes simulations from H_0 (true mean equal to 0) as example, but can be modified for simulations from H_1 with different true means.
 
4. quantile_example_g0.ipynb
File to reproduce the simulation for the g-and-k distribution model choice example. The file includes simulations from the model with g=0 (M1) as example, but can be modified for simulations from g != 0.

5. toad_example_RANDOM.ipynb
File to reproduce the simulation for toad movement model choice example. The file includes simulations from random return model (M1) example, but can be modified for simulations from nearest return model or the distance model.

Getting Started

Requirements

Python 3.9+

Jupyter Notebook or JupyterLab

Core packages:

pip install numpy pandas matplotlib seaborn scikit-learn scipy scipy.stats scipy.spatial.distance enum multiprocessing pprint random math time typing collection os pathlib functools

Running the notebooks

Clone the repository:

git clone https://github.com/username/abc-mc.git
cd abc-mc


Launch Jupyter:

jupyter notebook


Run the notebooks in order (expo_family_example_exp.ipynb, analysis.ipynb).

Results

The analysis notebook provides:

Code to save .csv files for model probabilities and parameter estimates for each method. 
Draw boxplots.

Reproducibility

Key random seeds are fixed where possible to ensure reproducibility.

Citation

If you use this code in your work, please cite:

@article{grazian2025approximate,
  title={Approximate Bayesian Computation with Statistical Distances for Model Selection},
  author={Grazian, Clara},
  journal={arXiv preprint arXiv:2410.21603},
  year={2025}
}

License

This project is released under the MIT License — see the LICENSE file for details.
