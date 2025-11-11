import os, sys, csv, numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Import abc_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def g_and_k_quantile(z, A=0, B=1, g=0, k=0, c=0.8):
    """Quantile function for the g-and-k distribution."""
    term = (1 + c * (1 - np.exp(-g * z)) / (1 + np.exp(-g * z)))
    return A + B * term * (1 + z**2)**k * z

# ========================
# QDA Simulation for ABC
# ========================
def simulate_for_observed(obs_data, obs_id, n_simulations):
    rng_obs = np.random.default_rng(obs_id)
    obs_labels = np.zeros(len(obs_data))

    def simulate_once(seed, obs_data):
        rng = np.random.default_rng(seed)
        dist_type = rng.integers(0, 2)  # 0 or 1

        n = len(obs_data)
        A, B, c = 0, 1, 0.8

        if dist_type == 0:
            g = 0.0
            k = rng.uniform(-0.5, 5)
        else:
            g = rng.uniform(0, 4)
            k = rng.uniform(-0.5, 5)

        z = rng.normal(0, 1, size=n)
        sim_data = g_and_k_quantile(z, A, B, g, k, c)

        obs_labels = np.zeros(n)
        sim_labels = np.ones(n)
        X = np.concatenate([obs_data, sim_data]).reshape(-1, 1)
        y = np.concatenate([obs_labels, sim_labels])

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        y_pred = qda.predict(X)
        acc = accuracy_score(y, y_pred)

        return acc, (g, k), dist_type

    results = Parallel(n_jobs=-1)(
        delayed(simulate_once)(i, obs_data) for i in range(n_simulations)
    )
    accuracies = np.array([r[0] for r in results])
    params = np.array([r[1] for r in results])  # array of (g,k)
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
        raise FileNotFoundError(f"Missing observed datasets at {observed_path}. Run g-and-k_example_distanceABC.py first.")

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
    qda_frequencies, mean_g_list, mean_k_list = [], [], []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        model_choices, g_values, k_values = [], [], []

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # ['obs_id', 'accuracy', 'g', 'k', 'dist_type']
            for row in reader:
                try:
                    g = float(row[2])
                    k = float(row[3])
                    model = int(row[4])
                    model_choices.append(model)
                    if model == 1:
                        g_values.append(g)
                        k_values.append(k)
                except (IndexError, ValueError):
                    continue

        # Compute frequency of Model 0 selections
        freq_model0 = model_choices.count(0) / len(model_choices) if model_choices else 0.0
        qda_frequencies.append(freq_model0)

        # Mean g and k for Model 1 draws
        mean_g = np.mean(g_values) if g_values else 0.0
        mean_k = np.mean(k_values) if k_values else 0.0
        mean_g_list.append(mean_g)
        mean_k_list.append(mean_k)

    # Save summary outputs
    np.savetxt(os.path.join(folder_path, 'gk_example_g0_QDA_probabilities.csv'),
               np.array(qda_frequencies), delimiter=',', header='QDA_Prob_Model0', comments='')
    np.savetxt(os.path.join(folder_path, 'gk_example_g0_QDA_mean_g.csv'),
               np.array(mean_g_list), delimiter=',', header='Mean_g_Model1', comments='')
    np.savetxt(os.path.join(folder_path, 'gk_example_g0_QDA_mean_k.csv'),
               np.array(mean_k_list), delimiter=',', header='Mean_k_Model1', comments='')

    print(f"Saved QDA summary results in {folder_path}")

if __name__ == '__main__':
    main()
