import os, sys, numpy as np
from multiprocessing import Pool, cpu_count

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import run_abc_for_one_observed, Distance, DISTANCE_LABELS

def g_and_k_quantile(z, A=0, B=1, g=0, k=0, c=0.8):
    term1 = (1 + c * (1 - np.exp(-g * z)) / (1 + np.exp(-g * z)))
    term2 = (1 + z**2)**k
    return A + B * term1 * term2 * z

def simulate_datasets(n_sim=10**6, sample_size=100):
    half = n_sim // 2
    A, B, c = 0, 1, 0.8
    k0 = np.random.uniform(-0.5, 5, size=half)
    g0 = np.zeros(half)
    models0 = np.zeros(half, dtype=int)
    z0 = np.random.normal(0, 1, size=(half, sample_size))
    sim0 = np.array([g_and_k_quantile(z0[i], A, B, g0[i], k0[i], c) for i in range(half)])

    g1 = np.random.uniform(0, 4, size=half)
    k1 = np.random.uniform(-0.5, 5, size=half)
    models1 = np.ones(half, dtype=int)
    z1 = np.random.normal(0, 1, size=(half, sample_size))
    sim1 = np.array([g_and_k_quantile(z1[i], A, B, g1[i], k1[i], c) for i in range(half)])

    sims = np.vstack((sim0, sim1))
    thetas = np.vstack((np.concatenate((g0, g1)), np.concatenate((k0, k1)))).T
    models = np.concatenate((models0, models1))
    return sims, thetas, models

def sample_g_and_k(n, A=0, B=1, c=0.8, g=0, k=2):
    z = np.random.normal(0, 1, size=n)
    return g_and_k_quantile(z, A, B, g, k, c)

def main():
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    DISTABC_DIR = os.path.join(RESULTS_DIR, 'distABC')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DISTABC_DIR, exist_ok=True)

    np.random.seed(42)
    n_observed, sample_size, n_sim = 100, 100, 10**6
    #n_observed, sample_size, n_sim = 2, 100, 10**3
    percentiles = [0.1, 0.05, 0.01]

    print("Generating observed datasets...")
    observed_datasets = np.array([sample_g_and_k(sample_size, g=0, k=2) for _ in range(n_observed)]) # Change this to g=1 for Model M1
    np.savez(os.path.join(DATA_DIR, 'observed_datasets.npz'), observed_datasets=observed_datasets)
    print(f" Saved observed datasets to {DATA_DIR}")

    sims, thetas, models = simulate_datasets(n_sim=n_sim, sample_size=sample_size)
    args_list = [(obs, sims, thetas, models, percentiles) for obs in observed_datasets]

    print(f"Running ABC with {cpu_count()} cores...")
    with Pool() as pool:
        all_results = pool.map(run_abc_for_one_observed, args_list)

    distance_names = list(DISTANCE_LABELS.values())
    n_dist, n_perc = len(distance_names), len(percentiles)
    prop_model_summary = np.zeros((n_observed, n_dist, n_perc))
    mean_theta_summary = np.zeros((n_observed, n_dist, n_perc, 2))

    for i, res in enumerate(all_results):
        for d_idx, dist_name in enumerate(distance_names):
            for p_idx, _ in enumerate(percentiles):
                # Extract dictionaries for this percentile
                model_probs = res[dist_name]['model_probs'][p_idx]
                theta_means = res[dist_name]['theta_means'][p_idx]

                # There are two models: model 0 (no skew) and model 1 (skewed)
                prop_model_summary[i, d_idx, p_idx] = model_probs.get(1, np.nan)
                mean_theta_summary[i, d_idx, p_idx, 0] = theta_means.get(0, np.nan)
                mean_theta_summary[i, d_idx, p_idx, 1] = theta_means.get(1, np.nan)


    np.savez(os.path.join(DISTABC_DIR, 'gk_example_g0_distanceABC_results.npz'),
             prop_model=prop_model_summary,
             mean_theta=mean_theta_summary,
             percentiles=percentiles,
             distance_names=distance_names)
    print(f" Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
