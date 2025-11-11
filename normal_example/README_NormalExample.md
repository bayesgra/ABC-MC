# README.md — Normal Example ABC Workflow

This project implements **Approximate Bayesian Computation (ABC)** for the toad movement example using three methods:
- **Distance-based ABC** — classical rejection ABC (creates observed datasets)
- **QDA-based ABC** — ABC via classification (through QDA)
- **SA-based ABC** — semiautomatic ABC
- **Neural Network** — multi-task deep learning approximation

All **observed datasets** are stored in `normal_example/data/`, while all **other results** are stored in `normal_example/results/`.

---

## Folder Structure
```
project_root/
│
├── abc_utils.py
│
├── normal_example/
│   ├── normal_example_distanceABC.py
│   ├── normal_example_qdaABC.py
│   ├── normal_example_NN.py
│   ├── data/
│   │   └── observed_datasets.npy
│   └── results/
│       ├── normal_example_m0.npz
│       ├── qdaABC/
│       │   ├── qda_simulations_obs_000.csv
│       │   ├── normal_model0_QDA_probabilities.csv
│       │   └── normal_model0_QDA_mean_mu.csv
│       ├── saABC/
│       │   ├── normal_model0_SA_probabilities.csv
│       │   └── normal_model0_SA_mean_mu.csv
│       └── NN/
│           ├── normal_model.h5
│           ├── NN_probabilities.csv
│           └── NN_params.csv
```

---

## Setup
Install dependencies (for example, using `pip`):
```bash
pip install numpy scipy scikit-learn tensorflow joblib pandas
```

---

## Workflow
### 1. **Distance-based ABC**
Generates synthetic data, **creates `data/observed_datasets.npy`**, and performs ABC comparisons using several distances (CvM, MMD, Wasserstein, Stat).
```bash
cd normal_example
python3 normal_example_distanceABC.py
```
**Outputs:**
- `data/observed_datasets.npy` — observed datasets for reuse.
- `results/distABC/normal_example_m0.npz` — ABC summary results.

---

### 2. **QDA-based ABC**
Uses QDA-ABC to perform the comparison with distanceABC.
```bash
python3 normal_example_qdaABC.py
```
**Reads:** `data/observed_datasets.npy`

**Outputs:**
- Individual simulation CSV files under `results/qdaABC/`
- Summary CSVs with posterior probabilities and mean parameter estimates.

---

### 3. **SA-based ABC**
Uses SA-ABC to perform the comparison with distanceABC.
```bash
R normal_example_saABC.R
```
**Reads:** `data/observed_datasets.npy`

**Outputs:**
- Individual simulation CSV files under `results/saABC/`
- Summary CSVs with posterior probabilities and mean parameter estimates.

---

### 4. **Neural Network**
Trains and evaluates a multi-task NN for simultaneous model and parameter estimation.

#### Training:
```bash
python3 normal_example_NN.py --train --epochs 10 --batch_size 32
```
This trains a multi-task NN to jointly predict model class and parameter values.

#### Prediction:
```bash
python3 normal_example_NN.py --predict
```
**Reads:** `data/observed_datasets.npy`

**Outputs:**
- `results/NN/normal_model.h5` — trained model.
- `results/NN/NN_probabilities.csv` — predicted model 0 probabilities.
- `results/NN/NN_params.csv` — estimated parameters.

---

## Notes
- All scripts are standalone.
- `distanceABC` is the only script that creates the `data/observed_datasets.npy` file (others only read it).
- You can re-run `distanceABC` to regenerate observed data if needed.

---

## Citation
If you use this workflow in research, please cite the underlying paper
Grazian, C (2025) "Approximate Bayesian Computation with Statistical Distances for Model Selection, arXiv:2410.21603

