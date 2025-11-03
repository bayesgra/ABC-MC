import os, sys, random, numpy as np
from multiprocessing import Pool, cpu_count

# Ensure abc_utils.py is accessible from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import run_abc_for_one_observed, Distance, DISTANCE_LABELS

# ========================
# Utilities
# ========================
def simulate_datasets(n_sim=10**3, sample_size=100, prior_mu_var=100):
    half = n_sim // 2
    mus0 = np.zeros(half)
    sim0 = np.random.normal(0, 1, size=(half, sample_size))
    mus1 = np.random.normal(0, prior_mu_var ** 0.5, size=half)
    sim1 = np.array([np.random.normal(mu, 1, size=sample_size) for mu in mus1])
    sims = np.vstack((sim0, sim1))
    mus = np.concatenate((mus0, mus1))
    models = np.concatenate((np.zeros(half, dtype=int), np.ones(half, dtype=int)))
    return sims, mus, models

def generate_observed_datasets(n_observed=100, sample_size=100, seed=123):
    np.random.seed(seed)
    return np.array([np.random.normal(0, 1, sample_size) for _ in range(n_observed)])

# ========================
# Main
# ========================
def main():
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample_size = 100
    n_sim = 10**6
    n_observed = 100
    percentiles = [0.1, 0.05, 0.01]

    sims, mus, models = simulate_datasets(n_sim=n_sim, sample_size=sample_size)

    observed_path = os.path.join(DATA_DIR, 'observed_datasets.npy')
    try:
        observed_datasets = np.load(observed_path, allow_pickle=True)
        print(f"Loaded observed datasets from {observed_path}")
    except FileNotFoundError:
        print("Generating new observed datasets...")
        observed_datasets = generate_observed_datasets(n_observed=n_observed, sample_size=sample_size)
        np.save(observed_path, observed_datasets)
        print(f"Saved observed datasets to {observed_path}")

    args_list = [(obs, sims, mus, models, percentiles) for obs in observed_datasets]

    print(f"Running ABC using {cpu_count()} cores...")
    with Pool() as pool:
        all_results = pool.map(run_abc_for_one_observed, args_list)

    distance_names = list(DISTANCE_LABELS.values())
    n_dist, n_perc = len(distance_names), len(percentiles)
    prop_model0_summary = np.zeros((n_observed, n_dist, n_perc))
    mean_mu_summary = np.zeros((n_observed, n_dist, n_perc))

    for i, res in enumerate(all_results):
        for d_idx, dist_name in enumerate(distance_names):
            prop_model0_summary[i, d_idx, :] = res[dist_name]['prop_model0']
            mean_mu_summary[i, d_idx, :] = res[dist_name]['mean_mu']

    output_file = os.path.join(RESULTS_DIR, 'normal_example_m0_distanceABC_results.npz')
    np.savez(output_file, prop_model0=prop_model0_summary, mean_mu=mean_mu_summary, percentiles=percentiles, distance_names=distance_names)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
