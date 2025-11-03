import os, sys, csv, numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Import abc_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ========================
# QDA Simulation for ABC
# ========================
def simulate_for_observed(obs_data, obs_id, n_simulations):
    rng_obs = np.random.default_rng(obs_id)
    obs_labels = np.zeros(len(obs_data))

    def simulate_once(seed):
        rng = np.random.default_rng(seed)
        dist_type = rng.integers(0, 3)
        if dist_type == 0:
            sim_data = np.random.exponential(scale=1.0, size=len(obs_data))
            param = np.mean(sim_data)
        elif dist_type == 1:
            mu = rng.normal(0, 1)
            sim_data = np.random.lognormal(mean=mu, sigma=1.0, size=len(obs_data))
            param = mu
        else:
            theta = rng.exponential(scale=1.0)
            sim_data = np.random.gamma(shape=2.0, scale=1/theta, size=len(obs_data))
            param = theta

        sim_labels = np.ones(len(sim_data))
        X = np.concatenate([obs_data, sim_data]).reshape(-1, 1)
        y = np.concatenate([obs_labels, sim_labels])

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        y_pred = qda.predict(X)
        acc = accuracy_score(y, y_pred)

        return acc, param, dist_type

    results = Parallel(n_jobs=-1)(delayed(simulate_once)(i) for i in range(n_simulations))
    accuracies = np.array([r[0] for r in results])
    params = np.array([r[1] for r in results])
    dist_types = np.array([r[2] for r in results])

    threshold = np.percentile(accuracies, 10)
    idx_selected = np.where(accuracies <= threshold)[0]
    print(f"Obs {obs_id:03d} — percentile threshold: {threshold:.4f} — {len(idx_selected)} rows selected")

    return [(obs_id, accuracies[i], params[i], dist_types[i]) for i in idx_selected]

# ========================
# Main function
# ========================
def main():
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_simulations = 10**6
    output_dir = os.path.join(RESULTS_DIR, 'qdaABC')
    os.makedirs(output_dir, exist_ok=True)

    observed_path = os.path.join(DATA_DIR, 'observed_datasets.npz')
    if not os.path.exists(observed_path):
        raise FileNotFoundError(f"Missing observed datasets at {observed_path}. Run expo_family_example_distanceABC.py first.")

    observed_datasets = np.load(observed_path, allow_pickle=True)['observed_datasets']
    print(f"Loaded {len(observed_datasets)} observed datasets.")

    for obs_id, obs_data in enumerate(observed_datasets):
        print(f"Running simulations for observed dataset {obs_id}")
        results = simulate_for_observed(obs_data, obs_id, n_simulations)

        filename = os.path.join(output_dir, f"qda_simulations_obs_{obs_id:03d}.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['obs_id', 'accuracy', 'param', 'dist_type'])
            writer.writerows(results)
        print(f"Saved {filename}")

    summarize_results(output_dir)

# ========================
# Summarize results
# ========================
def summarize_results(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    qda_frequencies = []
    qda_means = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        model_choices, parameters = [], []

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    param = float(row[2]); model = int(row[3])
                    model_choices.append(model); parameters.append((param, model))
                except (IndexError, ValueError):
                    continue

        freq_model0 = model_choices.count(0)/len(model_choices) if model_choices else 0.0
        qda_frequencies.append(freq_model0)
        model1_params = [p for p,m in parameters if m==1]
        mean_param_model1 = np.mean(model1_params) if model1_params else 0.0
        qda_means.append(mean_param_model1)

    np.savetxt(os.path.join(folder_path, 'expo_family_exp_QDA_probabilities.csv'), np.array(qda_frequencies), delimiter=',', header='QDA', comments='')
    np.savetxt(os.path.join(folder_path, 'expo_family_exp_QDA_params.csv'), np.array(qda_means), delimiter=',', header='QDA', comments='')
    print(f"Saved QDA summary results in {folder_path}")

if __name__ == '__main__':
    main()
