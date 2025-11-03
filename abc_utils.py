# abc_utils.py
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, cdist
from enum import Enum

# ========================
# Distance Enum
# ========================
class Distance(Enum):
    CVM = 0
    MMD = 1
    WASS = 2
    STAT = 3

DISTANCE_LABELS = {
    Distance.CVM: 'CvM',
    Distance.MMD: 'MMD',
    Distance.WASS: 'Wass',
    Distance.STAT: 'Stat'
}

# ========================
# Distance Functions
# ========================
def stat_distance(obs, sim):
    return np.abs(np.mean(obs) - np.mean(sim))

def cramer_von_mises_distance(obs, sim):
    n, m = len(obs), len(sim)
    combined = np.concatenate((obs, sim))
    ranks = rankdata(combined)
    obs_ranks, sim_ranks = np.sort(ranks[:n]), np.sort(ranks[n:])
    i, j = np.arange(1, n+1), np.arange(1, m+1)
    term1 = n * np.sum((obs_ranks - i)**2)
    term2 = m * np.sum((sim_ranks - j)**2)
    denom = n*m*(n+m)
    return (term1 + term2)/denom - (4*n*m - 1)/(6*(n+m))

def wasserstein_distance(obs, sim):
    if len(obs) == len(sim):
        return np.mean(np.abs(np.sort(obs) - np.sort(sim)))
    from scipy.stats import wasserstein_distance as ws
    return ws(obs, sim)

def gaussian_kernel(sq_distances, sigma):
    return np.exp(-sq_distances / (2 * sigma))

def maximum_mean_discrepancy(obs, sim, obs_sq_dist=None, sigma=None):
    if obs_sq_dist is None:
        obs_sq_dist = pdist(obs.reshape(-1,1), 'sqeuclidean')
    if sigma is None:
        sigma = np.median(obs_sq_dist)**0.5
    sim_sq_dist = pdist(sim.reshape(-1,1), 'sqeuclidean')
    mixed_sq_dist = cdist(obs.reshape(-1,1), sim.reshape(-1,1), 'sqeuclidean')
    k_xx = np.mean(gaussian_kernel(obs_sq_dist, sigma))
    k_yy = np.mean(gaussian_kernel(sim_sq_dist, sigma))
    k_xy = np.mean(gaussian_kernel(mixed_sq_dist, sigma))
    return k_xx + k_yy - 2 * k_xy

# ========================
# ABC Utility Functions
# ========================
def compute_distances(observed_sample, sims):
    n_sim = sims.shape[0]
    obs_sq_dist = pdist(observed_sample.reshape(-1,1), 'sqeuclidean')
    sigma = np.median(obs_sq_dist)**0.5
    distances = {name: np.zeros(n_sim) for name in DISTANCE_LABELS.values()}
    for i in range(n_sim):
        sim_sample = sims[i]
        distances['CvM'][i] = cramer_von_mises_distance(observed_sample, sim_sample)
        distances['Wass'][i] = wasserstein_distance(observed_sample, sim_sample)
        distances['MMD'][i] = maximum_mean_discrepancy(observed_sample, sim_sample, obs_sq_dist, sigma)
        distances['Stat'][i] = stat_distance(observed_sample, sim_sample)
    return distances

def summarize_percentile(mus, models, distances, percentile):
    n = len(distances)
    k = max(1, round(n * percentile / 100))
    indices = np.argsort(distances)[:k]
    selected_mus, selected_models = mus[indices], models[indices]
    return np.mean(selected_models == 0), np.mean(selected_mus)

def run_abc_for_one_observed(args):
    observed_sample, sims, mus, models, percentiles = args
    distances = compute_distances(observed_sample, sims)
    results = {}
    for dist_enum in Distance:
        dist_name = DISTANCE_LABELS[dist_enum]
        results[dist_name] = {'prop_model0': [], 'mean_mu': []}
        for perc in percentiles:
            prop0, mean_mu = summarize_percentile(mus, models, distances[dist_name], perc)
            results[dist_name]['prop_model0'].append(prop0)
            results[dist_name]['mean_mu'].append(mean_mu)
    return results

