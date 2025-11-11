import os, sys, random, numpy as np
from multiprocessing import Pool, cpu_count

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import run_abc_for_one_observed, Distance, DISTANCE_LABELS

def stat_distance(observed_data, simulated_data):
    log_observed = np.log(observed_data)
    observed_stats = np.array([np.sum(observed_data),
                               np.sum(log_observed),
                               np.sum(log_observed**2)])
    log_simulated = np.log(simulated_data)
    simulated_stats = np.array([np.sum(simulated_data),
                                np.sum(log_simulated),
                                np.sum(log_simulated**2)])
    return np.linalg.norm(observed_stats - simulated_stats)

def generate_observed_datasets(n_observed=100, sample_size=100, seed=42, save_path="data/observed_datasets.npz"):
    np.random.seed(seed)
    observed_datasets = np.random.exponential(scale=2.0, size=(n_observed, sample_size)) # Change this to lognormal or gamma distribution for Model M2 and M3
    np.savez(save_path, observed_datasets=observed_datasets)
    print(f" Saved {n_observed} observed datasets to '{save_path}'")
    return observed_datasets

def simulate_datasets(n_sim=10**6, sample_size=100):
    third = n_sim // 3
    theta0 = np.random.exponential(scale=1.0, size=third)
    sim0 = np.array([np.random.exponential(scale=1/theta, size=sample_size) for theta in theta0])
    models0 = np.zeros(third, dtype=int)
    theta1 = np.random.normal(0, 1, third)
    sim1 = np.array([np.random.lognormal(mean=theta, sigma=1, size=sample_size) for theta in theta1])
    models1 = np.ones(third, dtype=int)
    theta2 = np.random.exponential(scale=1.0, size=third)
    sim2 = np.array([np.random.gamma(shape=2, scale=1/theta, size=sample_size) for theta in theta2])
    models2 = np.full(third, 2, dtype=int)
    sims = np.vstack((sim0, sim1, sim2))
    thetas = np.concatenate((theta0, theta1, theta2))
    models = np.concatenate((models0, models1, models2))
    return sims, thetas, models

def main():
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    DISTABC_DIR = os.path.join(RESULTS_DIR, 'distABC')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DISTABC_DIR, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    n_observed, sample_size, n_sim = 100, 100, 10**6
    #n_observed, sample_size, n_sim = 2, 100, 10**3
    percentiles = [0.1, 0.05, 0.01]

    observed_datasets = generate_observed_datasets(
        n_observed=n_observed,
        sample_size=sample_size,
        save_path=os.path.join(DATA_DIR, 'observed_datasets.npz')
    )

    sims, thetas, models = simulate_datasets(n_sim=n_sim, sample_size=sample_size)
    args_list = [(obs, sims, thetas, models, percentiles) for obs in observed_datasets]

    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_abc_for_one_observed, args_list)

    distance_names = list(DISTANCE_LABELS.values())
    n_dist, n_perc, n_models = len(distance_names), len(percentiles), 3
    model_probs_summary = np.zeros((n_observed, n_dist, n_perc, n_models))
    theta_means_summary = np.zeros((n_observed, n_dist, n_perc, n_models))

    for i, res in enumerate(all_results):
        for d_idx, dist_name in enumerate(distance_names):
            for p_idx in range(n_perc):
                model_probs = res[dist_name]['model_probs'][p_idx]
                theta_means = res[dist_name]['theta_means'][p_idx]
                for m in range(n_models):
                    model_probs_summary[i, d_idx, p_idx, m] = model_probs.get(m, 0.0)
                    theta_means_summary[i, d_idx, p_idx, m] = theta_means.get(m, np.nan)

    np.savez(os.path.join(DISTABC_DIR, 'expofamily_example_exp_distanceABC_results.npz'),
             model_probs=model_probs_summary,
             theta_means=theta_means_summary,
             percentiles=percentiles,
             distance_names=distance_names,
             model_labels=["Expo", "LogN", "Gamma"])
    print(f"ABC results saved in {DISTABC_DIR}")

if __name__ == "__main__":
    main()

