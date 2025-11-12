import os
import numpy as np
import sys
from enum import Enum
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from toad_utils import *
import pandas as pd

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import *

def main():
    # ================================================================
    #  Folder setup
    # ================================================================
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    DISTABC_DIR = os.path.join(RESULTS_DIR, "distABC")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DISTABC_DIR, exist_ok=True)

    np.random.seed(42)

    # ================================================================
    #  Parameters
    # ================================================================
    n_observed = 100
    n_simulations = 100_000
    percentiles = [0.01, 0.005, 0.001]

    # ================================================================
    #  Step 1: Simulate training data
    # ================================================================
    print("Generating simulations...")
    sim_summaries, thetas, models = simulate_datasets_parallel(n_sim=n_simulations)

    # ---- Save simulated summaries and parameters ----
    sim_stats_path = os.path.join(DATA_DIR, "toad_simulated_stats.csv")
    sim_param_path = os.path.join(DATA_DIR, "toad_simulated_param.csv")

    pd.DataFrame(sim_summaries).to_csv(sim_stats_path, index=False)
    pd.DataFrame(thetas, columns=["alpha", "gamma", "p0", "d0"]).to_csv(sim_param_path, index=False)

    print(f"Saved simulated summaries to {sim_stats_path}")
    print(f"Saved simulated parameters to {sim_param_path}")

    # ================================================================
    #  Step 2: Generate or load observed datasets
    # ================================================================
    observed_path = os.path.join(DATA_DIR, "observed_datasets.npz")
    if os.path.exists(observed_path):
        observed_raw = np.load(observed_path, allow_pickle=True)["observed_datasets"]
        print(f"Loaded observed datasets: {observed_raw.shape}")
    else:
        print("Simulating new observed datasets...")
        observed_raw = [
            toad_movement_sample(Model.RANDOM, alpha=1.7, gamma=34, p0=0.6)
            for _ in range(n_observed)
        ]
        np.savez(observed_path, observed_datasets=np.array(observed_raw))
        print(f"Saved {len(observed_raw)} observed datasets to {observed_path}")

    # ---- Compute and save 48-d summaries for observed datasets ----
    def _flatten_summary(summary_dict, target_dim=48):
        vals = []
        def _rec(d):
            for v in d.values():
                if isinstance(v, dict):
                    _rec(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    vals.extend(np.ravel(v).tolist())
                else:
                    vals.append(v)
        _rec(summary_dict)
        arr = np.asarray(vals, dtype=float)
        if arr.size > target_dim:
            arr = arr[:target_dim]
        elif arr.size < target_dim:
            arr = np.pad(arr, (0, target_dim - arr.size), mode="constant", constant_values=np.nan)
        return arr

    obs_summ_list = []
    for obs in observed_raw:
        s = compute_displacement_summaries(obs)
        obs_summ_list.append(_flatten_summary(s, 48))

    obs_summ_df = pd.DataFrame(obs_summ_list, columns=[f"S{i+1}" for i in range(48)])
    obs_stats_path = os.path.join(DATA_DIR, "toad_observed_stats.csv")
    obs_summ_df.to_csv(obs_stats_path, index=False)
    print(f"Saved observed 48-dim summaries to {obs_stats_path}")

    # ================================================================
    #  Step 3: Run ABC in parallel
    # ================================================================
    print("Running ABC in parallel...")
    func = partial(
        run_one,
        sim_summaries=sim_summaries,
        sim_thetas=thetas,
        sim_models=models,
        percentiles=percentiles,
        output_dir=DISTABC_DIR,
    )

    with Pool(processes=cpu_count()) as pool:
        indexed_obs = list(enumerate(observed_raw))
        all_results = pool.map(func, indexed_obs)

    # ================================================================
    #  Step 4: Aggregate results
    # ================================================================
    distance_names = list(DISTANCE_LABELS.values())
    n_dist, n_perc = len(distance_names), len(percentiles)
    n_models = len(np.unique(models)) 
    
    model_probs_summary = np.zeros((n_observed, n_dist, n_perc, n_models))
    theta_means_summary = np.zeros((n_observed, n_dist, n_perc, n_models))
    
    for i, res in enumerate(all_results):
        for d_idx, dist_name in enumerate(distance_names):
            for p_idx in range(n_perc):
                model_probs = res[dist_name]["model_probs"][p_idx]
                theta_means = res[dist_name]["theta_means"][p_idx]
                for m in range(n_models):
                    model_probs_summary[i, d_idx, p_idx, m] = (
                        model_probs[m] if len(model_probs) > m else np.nan
                    )
                    theta_means_summary[i, d_idx, p_idx, m] = (
                        theta_means[m] if len(theta_means) > m else np.nan
                    )

    # ================================================================
    #  Step 5: Save results
    # ================================================================
    output_path = os.path.join(DISTABC_DIR, "toad_example_distanceABC_results.npz")
    np.savez(
        output_path,
        model_probs=model_probs_summary,
        theta_means=theta_means_summary,
        percentiles=percentiles,
        distance_names=distance_names,
        model_labels=["Random", "Nearest", "Distance"],
    )

    print(f"ABC results saved to: {output_path}")
    return model_probs_summary, theta_means_summary

if __name__ == "__main__":
    main()

