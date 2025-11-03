# README.md — Exponential Family Example Workflow

This project implements **Approximate Bayesian Computation (ABC)** for the toad movement example using three methods:
- **Distance-based ABC** — classical rejection ABC (creates observed datasets)
- **QDA-based ABC** — ABC via classification (through QDA)
- **SA-based ABC** — semiautomatic ABC
- **Neural Network** — multi-task deep learning approximation

---

## Folder Structure
```
project_root/
│
├── abc_utils.py
│
└── expo_family_example/
    ├── expo_family_example_distanceABC.py
    ├── expo_family_example_qdaABC.py
    ├── expo_family_example_NN.py
    ├── data/
    │   └── observed_datasets.npz
    └── results/
        ├── expo_family_results.npz
        ├── qdaABC/
        │   ├── qda_simulations_obs_000.csv
        │   ├── expo_family_QDA_probabilities.csv
        │   └── expo_family_QDA_mean_theta.csv
        └── NN/
            ├── expo_family_model.h5
            ├── expo_family_NN_probabilities.csv
            └── expo_family_NN_params.csv
```

---

## Setup
Install dependencies:
```bash
pip install numpy scipy scikit-learn tensorflow joblib pandas
```

---

## Workflow

### 1. **Distance-based ABC**
Generates observed datasets and performs ABC simulations.
```bash
cd expo_family_example
python expo_family_example_distanceABC.py
```
**Outputs:**
- `data/observed_datasets.npz` — simulated observed datasets
- `results/expo_family_results.npz` — ABC posterior summaries

---

### 2. **QDA-based ABC**
Uses QDA-ABC to perform the comparison with distanceABC.
```bash
python expo_family_example_qdaABC.py
```
**Reads:** `data/observed_datasets.npz`

**Outputs:**
- `results/qdaABC/expo_family_QDA_probabilities.csv`
- `results/qdaABC/expo_family_QDA_mean_theta.csv`

---

### 3. **SA-based ABC**
Uses SA-ABC to perform the comparison with distanceABC.
```bash
R expo_family_example_saABC.R
```
**Reads:** `data/observed_datasets.npz`

**Outputs:**
- `results/saABC/expo_family_SA_probabilities.csv`
- `results/saABC/expo_family_SA_mean_theta.csv`

---

### 4. **Neural Network**
Trains and evaluates a multi-task NN for simultaneous model and parameter estimation.

#### Train:
```bash
python expo_family_example_NN.py --train --epochs 10 --batch_size 32
```
#### Predict:
```bash
python expo_family_example_NN.py --predict
```
**Reads:** `data/observed_datasets.npz`

**Outputs:**
- `results/NN/expo_family_model.h5` — trained model
- `results/NN/expo_family_NN_probabilities.csv` — predicted model probabilities
- `results/NN/expo_family_NN_params.csv` — estimated parameters

---

## Notes
- Only `expo_family_example_distanceABC.py` creates the observed dataset in `data/`.
- The other scripts only read from `data/` and write their results to `results/`.
- All scripts can be run independently but rely on consistent folder organization.

---

## Citation
If you use this workflow in research, please cite the underlying paper
Grazian, C (2025) "Approximate Bayesian Computation with Statistical Distances for Model Selection, arXiv:2410.21603

