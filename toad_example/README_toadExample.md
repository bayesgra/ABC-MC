# README.md — toad movement ABC Workflow

This project implements **Approximate Bayesian Computation (ABC)** for the toad movement example using three methods:
- **Distance-based ABC** — classical rejection ABC (creates observed datasets)
- **QDA-based ABC** — ABC via classification (through QDA)
- **SA-based ABC** — semiautomatic ABC
- **Neural Network** — multi-task deep learning approximation

All scripts interact with two shared directories:
- `data/` — where observed datasets are stored
- `results/` — where each script saves its output results

---

##  Folder Structure
```
project_root/
│
├── abc_utils.py
│
└── toad_example/
    ├── toad_example_distanceABC.py
    ├── toad_example_qdaABC.py
    ├── toad_example_NN.py
    ├── data/
    │   └── observed_datasets.npz
    └── results/
        ├── toad_example_results.npz
        ├── qdaABC/
        │   ├── qda_simulations_obs_000.csv
        │   ├── toad_QDA_probabilities.csv
        │   └── toad_QDA_params.csv
        ├── saABC/
        │   ├── toad_QDA_probabilities.csv
        │   └── toad_QDA_params.csv
        └── NN/
            ├── toad_model.h5
            ├── toad_NN_probabilities.csv
            └── toad_NN_params.csv
```

---

##  Setup
To install dependencies:
```bash
pip install numpy scipy scikit-learn tensorflow joblib pandas
```

Ensure that `abc_utils.py` is located **one directory above** the `toad_example` folder.
Ensure that `toad_utils.py` is located in the `toad_example` folder.

---

##  Workflow

### 1. **Distance-based ABC**
Generates observed datasets and performs ABC comparisons using several distances (CvM, MMD, Wasserstein, Stat).
```bash
cd toad_example
python toad_example_distanceABC.py
```
**Outputs:**
- `data/observed_datasets.npz` — simulated observed datasets
- `results/toad_example_results.npz` — ABC posterior summaries

---

### 2. **QDA-based ABC**
Uses QDA-ABC to perform the comparison with distanceABC.
```bash
python toad_example_qdaABC.py
```
**Reads:** `data/observed_datasets.npz`

**Outputs:**
- `results/qdaABC/toad_QDA_probabilities.csv`
- `results/qdaABC/toad_QDA_mean_theta.csv`

---

### 3. **SA-based ABC**
Uses SA-ABC to perform the comparison with distanceABC.
```bash
R toad_example_saABC.R
```
**Reads:** `data/observed_datasets.npy`

**Outputs:**
- Individual simulation CSV files under `results/saABC/`
- Summary CSVs with posterior probabilities and mean parameter estimates.

---

### 4. **Neural Network**
Trains and evaluates a multi-task NN for simultaneous model and parameter estimation.

#### Train:
```bash
python toad_example_NN.py --train --epochs 10 --batch_size 32
```
#### Predict:
```bash
python toad_example_NN.py --predict
```
**Reads:** `data/observed_datasets.npz`

**Outputs:**
- `results/NN/toad_model.h5` — trained NN model
- `results/NN/toad_NN_probabilities.csv` — predicted model probabilities
- `results/NN/toad_NN_params.csv` — predicted parameters

---

## Notes
- Only `toad_example_distanceABC.py` creates the observed datasets in `data/`.
- The other scripts only read from `data/` and write to `results/`.
- Each script can be executed independently.

---

## Citation
If you use this workflow in research, please cite the underlying paper
Grazian, C (2025) "Approximate Bayesian Computation with Statistical Distances for Model Selection, arXiv:2410.21603

