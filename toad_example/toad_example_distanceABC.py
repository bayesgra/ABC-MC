import os
import numpy as np
from enum import Enum
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import toad_utils
import pandas as pd

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import abc_utils

# -------------------------
# Simulation helpers
# -------------------------
def simulate_one(model: Model):
    # sample parameters from broad priors
    alpha = np.random.uniform(1.0, 2.0)
    gamma = np.random.uniform(10.0, 100.0)
    p0    = np.random.uniform(0.0, 1.0)
    if model == Model.DISTANCE:
        d0 = np.random.uniform(20.0, 2000.0)
        Y = toad_movement_sample(model, alpha, gamma, p0, d0)
        theta = (alpha, gamma, p0, d0)
    else:
        Y = toad_movement_sample(model, alpha, gamma, p0)
        theta = (alpha, gamma, p0, np.nan)
    summary = compute_displacement_summaries(Y)
    return summary, theta, model.value

def simulate_datasets_parallel(n_sim: int, processes: int | None = None):
    if processes is None:
        processes = cpu_count()
    # balance across models (RANDOM, NEAREST, DISTANCE)
    n_each = n_sim // 3  # <— safe integer split
    model_list = ([Model.RANDOM] * n_each +
                  [Model.NEAREST] * n_each +
                  [Model.DISTANCE] * n_each)

    with Pool(processes) as pool:
        results = pool.map(simulate_one, model_list)

    summaries, thetas, models = zip(*results)
    return np.array(summaries), np.array(thetas), np.array(models)

# ----------------------------------------
# ABC on a single observed dataset (RANDOM)
# ----------------------------------------
def summarize_percentile(sim_thetas, sim_models, dists, percentile=1.0):
    dists = np.asarray(dists)
    sim_thetas = np.asarray(sim_thetas)
    sim_models = np.asarray(sim_models)

    # additional guard for empty inputs
    if dists.size == 0 or sim_thetas.size == 0 or sim_models.size == 0:
        return np.full(3, np.nan), np.full(sim_thetas.shape[1] if sim_thetas.ndim == 2 else 1, np.nan)

    thr = np.percentile(dists, percentile * 100.0)
    idx = np.where(dists <= thr)[0]

    if idx.size == 0:
        num_models = len(np.unique(sim_models))
        num_thetas = sim_thetas.shape[1] if sim_thetas.ndim > 1 else 1
        return np.full(num_models, np.nan), np.full(num_thetas, np.nan)

    acc_models = sim_models[idx]
    acc_thetas = sim_thetas[idx]

    unique_m, counts = np.unique(acc_models, return_counts=True)
    model_probs = np.zeros(len(np.unique(sim_models)))
    model_probs[unique_m] = counts / counts.sum()

    theta_means = np.nanmean(acc_thetas, axis=0)
    return model_probs, theta_means

def run_one_observed(i, obs_data, sim_summaries, sim_thetas, sim_models, percentiles, output_dir: Path):
    # summaries and distances
    obs_summary   = compute_displacement_summaries(obs_data)
    lag_distances = compute_lag_distances(obs_summary, sim_summaries)
    combined_all  = combine_distances(lag_distances, omega=0.2)

    results = {}
    for dist_enum in Distance:
        dist_name = DISTANCE_LABELS[dist_enum]
        combined_dist = combined_all[dist_name]

        model_probs_list = []
        theta_means_list = []
        for perc in percentiles:
            mprob, tmean = summarize_percentile(sim_thetas, sim_models, combined_dist, perc)
            model_probs_list.append(np.array(mprob[:3]) if mprob is not None else np.full(3, np.nan))
            # theta: (alpha, gamma, p0, d0)
            theta_means_list.append(np.array(tmean[:4]) if tmean is not None else np.full(4, np.nan))

        results[dist_name] = {
            "model_probs": model_probs_list,  # shape (n_perc, 3)
            "theta_means": theta_means_list   # shape (n_perc, 4)
        }

    # model-aware filename (random observed set)
    out_path = output_dir / f"toad_result_random_{i+1}.npz"
    np.savez(out_path, result=results, index=i, percentiles=np.array(percentiles))
    return results

# -----------
# Main driver
# -----------
def main():
    np.random.seed(42)

    # I/O
    base_dir    = Path(__file__).parent
    data_dir    = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_dir  = results_dir / "random"  # use model-aware subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_path = data_dir / "observed_datasets.npz"
    if not observed_path.exists():
        print("Generating new observed datasets...")
        n_observed = 100
        observed_datasets = []
        for _ in range(n_observed):
            # Choose one model for the observed data, e.g., RANDOM
            obs = toad_movement_sample(
                model=Model.RANDOM,
                alpha=1.7, gamma=34.0, p0=0.6, d0=0.0,
                num_toads=66, num_days=63
            )
            observed_datasets.append(obs)

    observed_datasets = np.stack(observed_datasets)
    np.savez(observed_path, observed_datasets=observed_datasets)
    print(f"Saved {n_observed} observed datasets to {observed_path}")

    # Load observed datasets (RANDOM model observed set, one model per call)
    data = np.load(observed_path, allow_pickle=True)
    observed_raw = data["observed_datasets"]  # shape (n_observed, num_days, num_toads)
    n_observed = observed_raw.shape[0]
    print(f"Loaded observed datasets: {observed_raw.shape}")

    # Simulations used across ALL candidate models (RANDOM/NEAREST/DISTANCE)
    n_simulations = 100_000
    percentiles   = [0.01, 0.005, 0.001]

    print("Generating simulations across all models...")
    sim_summaries, sim_thetas, sim_models = simulate_datasets_parallel(n_simulations)

    # ------------------------------------------------------------
    # Save simulated summaries and corresponding parameters
    # ------------------------------------------------------------

    # Flatten model labels
    model_labels = sim_models.astype(int)

    # Save summary statistics (48 columns)
    stats_df = pd.DataFrame(sim_summaries, columns=[f"stat_{i+1}" for i in range(sim_summaries.shape[1])])
    stats_df["model"] = model_labels
    stats_path = output_dir / "toad_simulated_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Save corresponding parameters
    param_df = pd.DataFrame(sim_thetas, columns=["alpha", "gamma", "p0", "d0"])
    param_df["model"] = model_labels
    param_path = output_dir / "toad_simulated_param.csv"
    param_df.to_csv(param_path, index=False)

    print(f"Saved simulated summaries to {stats_path}")
    print(f"Saved simulated parameters to {param_path}")

    # Run ABC in parallel over the observed set
    print("Running ABC over observed datasets...")
    func = partial(
        run_one_observed,
        sim_summaries=sim_summaries,
        sim_thetas=sim_thetas,
        sim_models=sim_models,
        percentiles=percentiles,
        output_dir=output_dir
    )

    with Pool(processes=cpu_count()) as pool:
        all_results = pool.starmap(func, [(i, obs) for i, obs in enumerate(observed_raw)])

    # Aggregate (same layout you already use elsewhere)
    distance_names = list(DISTANCE_LABELS.values())
    n_dist = len(distance_names)
    n_perc = len(percentiles)

    model_probs_summary = np.zeros((n_observed, n_dist, 3, n_perc))
    theta_means_summary = np.zeros((n_observed, n_dist, 4, n_perc))

    for i, result in enumerate(all_results):
        for j, dist_name in enumerate(distance_names):
            model_probs = np.array(result[dist_name]["model_probs"])  # (n_perc, 3)
            theta_means = np.array(result[dist_name]["theta_means"])  # (n_perc, 4)
            model_probs_summary[i, j] = model_probs.T
            theta_means_summary[i, j] = theta_means.T

    # Save aggregate
    np.save(output_dir / "model_probs_summary.npy", model_probs_summary)
    np.save(output_dir / "theta_means_summary.npy", theta_means_summary)
    print(f"Saved summaries in {output_dir}")

    return model_probs_summary, theta_means_summary

if __name__ == "__main__":
    _ = main()

