import numpy as np
import sys
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, cdist
from enum import Enum
from multiprocessing import Pool, cpu_count
import pprint
import os

import random
import math
import time
from scipy.stats import levy_stable
from typing import List, Tuple

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import *

# -------------------------
# ABC with Summary Statistics
# -------------------------
# Custom statistical distance (e.g., sum of log quantile differences)
def statistical_distance(x, y):
    q = np.linspace(0.0, 1.0, 11)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    with np.errstate(divide='ignore'):
        logdiff = np.abs(np.log1p(xq) - np.log1p(yq))
    return np.nansum(logdiff)

# -------------------------
# Define toad movement models
# -------------------------
class Model(Enum):
    RANDOM = 0
    NEAREST = 1
    DISTANCE = 2

# -------------------------
# Simulate from toad movement models
# -------------------------

def distance_based_probs(position: float, refuge_locations: np.ndarray, p0: float, d0: float) -> np.ndarray:
    # Calculating individual return probabilties based on the current position compared to the 
    # refuge locations for the distance-based return model
    refuge_distances = np.abs(position - refuge_locations)

    return p0 * np.exp(-refuge_distances / d0)

def toad_movement_sample(model, alpha: float, gamma: float, p0: float, d0: float = None,
                         num_toads: int = 66, num_days: int = 63) -> np.ndarray:
    """
    Simulate toad movement trajectories under three models:
      - RANDOM: independent random movement
      - NEAREST: toads may return to their nearest past location
      - DISTANCE: toads may return to a past refuge with distance-based probability
    """

    # --- Allow string or Enum as model input ---
    if isinstance(model, str):
        model_str = model.lower()
        if model_str == "random":
            model = Model.RANDOM
        elif model_str == "nearest":
            model = Model.NEAREST
        elif model_str == "distance":
            model = Model.DISTANCE
        else:
            raise ValueError(f"Unknown model string: {model}")

    # --- Initialize state arrays ---
    toad_positions = np.zeros((num_days, num_toads))
    refuge_counts = np.ones(num_toads, dtype=int)
    refuge_locations = np.zeros((num_days, num_toads))
    refuge_probs = None

    # --- Model-specific setup ---
    if model == Model.DISTANCE:
        # Prepare refuge probabilities based on distance decay (using d0)
        distances = np.arange(num_days)
        refuge_probs = np.exp(-distances / (d0 if d0 is not None else 100.0))
        refuge_probs = np.tile(refuge_probs[:, None], (1, num_toads))
    else:
        no_return_probs = 1 - p0

    # --- Simulation loop ---
    for i in range(1, num_days):
        # Which toads return today?
        return_mask = np.random.rand(num_toads) < p0
        no_return_mask = ~return_mask

        # Generate new candidate positions
        new_pos = np.random.normal(toad_positions[i - 1], gamma)

        if model == Model.RANDOM:
            # Fully random step
            toad_positions[i] = new_pos

        elif model == Model.NEAREST:
            # Each returning toad goes back to its nearest past location
            return_idx = np.where(return_mask)[0]
            for j in return_idx:
                past_positions = toad_positions[:i, j]
                nearest_idx = np.argmin(np.abs(new_pos[j] - past_positions))
                toad_positions[i, j] = past_positions[nearest_idx]
            # Non-returning toads move randomly
            toad_positions[i, no_return_mask] = new_pos[no_return_mask]

        elif model == Model.DISTANCE:
            # Distance-weighted return to past refuges
            return_idx = np.where(return_mask)[0]
            for j in return_idx:
                n_choices = int(refuge_counts[j])
                if n_choices < 1:
                    n_choices = 1
                p = refuge_probs[:n_choices, j] / np.sum(refuge_probs[:n_choices, j])
                chosen = np.random.choice(np.arange(n_choices), p=p)
                toad_positions[i, j] = refuge_locations[chosen, j]

            # Non-returning toads establish new refuges
            refuge_locations[refuge_counts[no_return_mask].astype(int), no_return_mask] = new_pos[no_return_mask]
            refuge_counts[no_return_mask] += 1

    return toad_positions

# -------------------------
# Compute summaries (48 per dataset): this is the final dataset
# -------------------------
def compute_displacement_summaries(Y: np.ndarray, lags=[1,2,4,8], threshold=10.0):
    summaries = []
    n_days, n_toads = Y.shape
    for lag in lags:
        displacements = np.abs(Y[lag:, :] - Y[:-lag, :]).flatten()
        returns = np.sum(displacements < threshold)
        non_returns = displacements[displacements >= threshold]
        summaries.append({'returns': returns, 'non_returns': non_returns})
    # Convert list to dict keyed by lag for consistency
    return dict(zip(lags, summaries))


# -------------------------
# Compute distances depending on lags
# -------------------------
def compute_lag_distances(obs_summary, sim_summaries, omega=0.2):
    """
    obs_summary: dict with lag keys (1,2,4,8), each maps to dict with 'returns' and 'non_returns'
    sim_summaries: list of such dicts, one per simulation
    Returns: list of dicts, each with keys 'CvM', 'Wass', 'MMD', 'Stat', each containing 'return' and 'non_return'
    """
    lag_keys = [1, 2, 4, 8]
    n_sim = len(sim_summaries)
    all_results = []

    for sim_idx, sim_summary in enumerate(sim_summaries):
        abs_diff_sum = 0
        cvm_vals = []
        wass_vals = []
        mmd_vals = []
        stat_vals = []

        for lag in lag_keys:
            ret_obs = obs_summary[lag]['returns']
            ret_sim = sim_summary[lag]['returns']
            nonret_obs = obs_summary[lag]['non_returns']
            nonret_sim = sim_summary[lag]['non_returns']

            abs_diff_sum += abs(ret_obs - ret_sim)

            if len(nonret_obs) > 0 and len(nonret_sim) > 0:
                cvm_vals.append(cramer_von_mises_distance(nonret_obs, nonret_sim))
                wass_vals.append(wasserstein_distance(nonret_obs, nonret_sim))
                mmd_vals.append(maximum_mean_discrepancy(nonret_obs, nonret_sim))
                stat_vals.append(statistical_distance(nonret_obs, nonret_sim))
            else:
                cvm_vals.append(np.nan)
                wass_vals.append(np.nan)
                mmd_vals.append(np.nan)
                stat_vals.append(np.nan)

        sim_result = {
            'CvM': {'return': abs_diff_sum, 'non_return': np.nanmean(cvm_vals)},
            'Wass': {'return': abs_diff_sum, 'non_return': np.nanmean(wass_vals)},
            'MMD': {'return': abs_diff_sum, 'non_return': np.nanmean(mmd_vals)},
            'Stat': {'return': abs_diff_sum, 'non_return': np.nanmean(stat_vals)}
        }

        all_results.append(sim_result)

    return all_results

# -------------------------
# Combine distances
# -------------------------
def combine_distances(dist_results, omega=0.2):
    """
    Combine distances per simulation with weighting and normalization.

    Parameters:
        dist_results: list of dicts, one per simulation, each with metrics and 'return'/'non_return'
        omega: float weight for return distances

    Returns:
        combined_distances: np.array of combined distance per simulation
    """
    n_sim = len(dist_results)
    metrics = dist_results[0].keys()

    combined_per_metric = {}

    for metric in metrics:
        # Extract arrays of distances
        ret_dists = np.array([d[metric]['return'] for d in dist_results])
        nonret_dists = np.array([d[metric]['non_return'] for d in dist_results])

        # Avoid division by zero
        max_ret = ret_dists.max() if ret_dists.max() > 0 else 1e-10
        max_nonret = nonret_dists.max() if nonret_dists.max() > 0 else 1e-10

        # Weighted normalized distance per simulation
        combined = omega * (ret_dists / max_ret) + (1 - omega) * (nonret_dists / max_nonret)
        combined_per_metric[metric] = combined

    return combined_per_metric

# -------------------------
# Simulate datasets for the ABC run
# -------------------------
def simulate_one(args):
    model = args
    alpha = np.random.uniform(1, 2.0)
    gamma = np.random.uniform(10, 100)
    p0 = np.random.uniform(0, 1)

    if model == Model.DISTANCE:
        d0 = np.random.uniform(20, 2000)
        Y = toad_movement_sample(model, alpha, gamma, p0, d0)
        thetas = (alpha, gamma, p0, d0)
    else:
        Y = toad_movement_sample(model, alpha, gamma, p0)
        thetas = (alpha, gamma, p0, np.nan)   

    summary = compute_displacement_summaries(Y)
    return summary, thetas, model.value

# -------------------------
# Simulate in parallel the ABC run
# -------------------------
def simulate_datasets_parallel(n_sim: int, processes: int = None):
    if processes is None:
        processes = cpu_count()

    # Create a model list with roughly equal representation
    models_all = [Model.RANDOM, Model.NEAREST, Model.DISTANCE]
    base_n = n_sim // len(models_all)
    remainder = n_sim % len(models_all)

    model_list = []
    for i, m in enumerate(models_all):
        count = base_n + (1 if i < remainder else 0)
        model_list.extend([m] * count)

    print(f"Simulating {len(model_list)} datasets: "
          f"{model_list.count(Model.RANDOM)} random, "
          f"{model_list.count(Model.NEAREST)} nearest, "
          f"{model_list.count(Model.DISTANCE)} distance")

    with Pool(processes) as pool:
        results = pool.map(simulate_one, model_list)

    summaries, thetas, models = zip(*results)
    return np.array(summaries), np.array(thetas), np.array(models)

# -------------------------
# Summarize percentile for toad movement example
# -------------------------
from collections import defaultdict

def summarize_percentile(sim_thetas, sim_models, dists, percentile=1.0):
    dists = np.asarray(dists)
    sim_thetas = np.asarray(sim_thetas)
    sim_models = np.asarray(sim_models)

    # Determine threshold
    threshold = np.percentile(dists, percentile * 100)
    accepted_idx = np.where(dists <= threshold)[0]

    # Log how many are accepted
    print(f"Percentile {percentile:.2f}: accepted {len(accepted_idx)} / {len(dists)} simulations")

    # -------------------------------------------------------------
    # If nothing accepted, return explicit arrays for 3 models
    # -------------------------------------------------------------
    if len(accepted_idx) == 0:
        num_thetas = sim_thetas.shape[1] if sim_thetas.ndim > 1 else 1
        model_probs = np.zeros(3)
        theta_means = np.full((3, num_thetas), np.nan)
        return model_probs, theta_means
    # -------------------------------------------------------------

    accepted_models = sim_models[accepted_idx]
    accepted_thetas = sim_thetas[accepted_idx]

    # -------------------------------------------------------------
    # Compute model probabilities for all 3 models explicitly
    # -------------------------------------------------------------
    model_probs = np.zeros(3)
    unique_models, counts = np.unique(accepted_models, return_counts=True)
    for um, c in zip(unique_models, counts):
        if um < 3:  # safeguard in case of unexpected IDs
            model_probs[int(um)] = c / counts.sum()
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Compute mean theta per model, pad missing models with NaNs
    # -------------------------------------------------------------
    num_thetas = sim_thetas.shape[1] if sim_thetas.ndim > 1 else 1
    theta_means = np.full((3, num_thetas), np.nan)
    for m in range(3):
        idx_m = np.where(accepted_models == m)[0]
        if len(idx_m) > 0:
            theta_means[m] = np.nanmean(accepted_thetas[idx_m], axis=0)
    # -------------------------------------------------------------

    return model_probs, theta_means
    
# -------------------------
# Function to run one ABC run for one observed dataset
# -------------------------
def run_abc_for_one_observed(i, obs_data, sim_summaries, sim_thetas, sim_models, percentiles, output_dir):
    """
    Run ABC for a single observed dataset and save results.
    Guarantees that for each distance, lists have len(percentiles) entries,
    even if no simulations are accepted.
    """
    obs_summary = compute_displacement_summaries(obs_data)
    lag_distances = compute_lag_distances(obs_summary, sim_summaries)
    combined_all = combine_distances(lag_distances, omega=0.2)

    results = {}

    for dist_enum in Distance:
        dist_name = DISTANCE_LABELS[dist_enum]
        combined_dist = combined_all[dist_name]

        model_probs_list = []
        theta_means_list = []

        for perc in percentiles:
            model_probs = None
            theta_means = None
            try:
                model_probs, theta_means = summarize_percentile(sim_thetas, sim_models, combined_dist, perc)
            except Exception as e:
                print(f"[Warning] summarize_percentile failed for {dist_name}, perc={perc}: {e}")

            # ---- Always append a result for this percentile ----
            if model_probs is None or theta_means is None:
                model_probs_arr = np.full(3, np.nan)
                theta_means_arr = np.full(3, np.nan)
            else:
                # Handle dicts or arrays safely
                if isinstance(model_probs, dict):
                    model_probs_arr = np.array([model_probs.get(m, 0.0) for m in range(3)])
                else:
                    model_probs_arr = np.array(model_probs).reshape(-1)[:3]

                if isinstance(theta_means, dict):
                    theta_means_arr = np.array([theta_means.get(m, np.nan) for m in range(3)])
                else:
                    theta_means_arr = np.array(theta_means).reshape(-1)[:3]

            model_probs_list.append(model_probs_arr)
            theta_means_list.append(theta_means_arr)
            # ----------------------------------------------------

        # Ensure correct number of entries even if loop fails early
        while len(model_probs_list) < len(percentiles):
            model_probs_list.append(np.full(3, np.nan))
            theta_means_list.append(np.full(3, np.nan))

        results[dist_name] = {
            "model_probs": model_probs_list,
            "theta_means": theta_means_list,
        }

    # === Save to disk ===
    out_path = Path(output_dir) / f"toad_result_random_{i+1}.npz"
    np.savez(
        out_path,
        result=results,
        index=i,
        percentiles=percentiles
    )

    # Debug: confirm lengths
    for dn, val in results.items():
        if len(val["model_probs"]) != len(percentiles):
            print(f"[Warning] {dn}: expected {len(percentiles)} entries, got {len(val['model_probs'])}")

    return results


import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from functools import partial
import numpy as np

def run_one(indexed_obs, sim_summaries, sim_thetas, sim_models, percentiles, output_dir):
    i, obs = indexed_obs
    return run_abc_for_one_observed(i, obs, sim_summaries, sim_thetas, sim_models, percentiles, output_dir)

