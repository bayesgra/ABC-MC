# ================================================================
# Shared simulation and helper functions for Toad ABC examples
# ================================================================

import numpy as np
from enum import Enum
from scipy.stats import levy_stable


# ================================================================
# Enums and constants
# ================================================================

class Model(Enum):
    RANDOM = 0
    NEAREST = 1
    DISTANCE = 2


# ================================================================
# Probability and movement functions
# ================================================================

def distance_based_probs(position, refuge_locations, p0=0.3, d0=1.0):
    """
    Compute probability of moving toward each refuge based on distance.
    """
    refuge_distances = np.abs(position - refuge_locations)
    probs = p0 * np.exp(-refuge_distances / d0)
    probs /= np.sum(probs)
    return probs


def toad_movement_sample(model="random", alpha=1.5, gamma=0.2, p0=0.3,
                         d0=1.0, num_toads=66, num_days=63, seed=None):
    """
    Simulate movement distances for a set of toads.

    Parameters
    ----------
    model : str
        Movement model type ("random", "nearest", "distance").
    alpha, gamma, p0, d0 : float
        Model parameters.
    num_toads, num_days : int
        Number of toads and observation days.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated daily displacements (num_toads × num_days).
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_toads, num_days))
    refuge_locations = np.linspace(0, 100, 10)

    for t in range(1, num_days):
        if model == "random":
            step = levy_stable.rvs(alpha, gamma, size=num_toads, random_state=rng)
        elif model == "distance":
            probs = np.array([
                distance_based_probs(positions[i, t-1], refuge_locations, p0, d0)
                for i in range(num_toads)
            ])
            targets = [rng.choice(refuge_locations, p=p) for p in probs]
            step = np.array(targets) - positions[:, t-1]
        else:  # nearest refuge
            nearest = np.array([
                refuge_locations[np.argmin(np.abs(positions[i, t-1] - refuge_locations))]
                for i in range(num_toads)
            ])
            step = nearest - positions[:, t-1]

        positions[:, t] = positions[:, t-1] + step

    return positions


# ================================================================
# Summary statistics
# ================================================================

import numpy as np

def compute_displacement_summaries(Y: np.ndarray, lags=(1, 2, 4, 8), return_threshold=10.0):
    """
    Compute 48 summary statistics for a single toad simulation matrix Y (num_days x num_toads).

    For each lag in {1, 2, 4, 8}:
      - Compute displacements |Y_{i+lag, j} - Y_{i,j}|
      - Count returns: displacements <= 10
      - For non-returns (>=10), compute deciles (0.0, 0.1, ..., 1.0)
      - Take 11 log-differences of consecutive deciles
      => 12 stats per lag (1 return count + 11 decile log-diffs)
    Returns
    -------
    np.ndarray of shape (48,)
    """
    stats = []

    for lag in lags:
        # displacements for this lag
        disp = np.abs(Y[lag:, :] - Y[:-lag, :]).ravel()

        # return / non-return split
        is_return = disp <= return_threshold
        n_return = np.sum(is_return)
        non_returns = disp[~is_return]

        if non_returns.size < 2:
            # not enough data to compute quantiles
            qdiffs = np.full(11, np.nan)
        else:
            # quantiles at 0, 0.1, ..., 1
            qs = np.linspace(0, 1, 11)
            quantiles = np.quantile(non_returns, qs)
            # consecutive log differences, safe against zeros
            qdiffs = np.full(11, np.nan)
            for i in range(len(quantiles) - 1):
                diff = quantiles[i + 1] - quantiles[i]
                qdiffs[i] = np.log(diff) if diff > 0 else np.nan

        stats.extend([n_return] + qdiffs.tolist())

    return np.array(stats, dtype=float)


def get_statistics(simulations: np.ndarray):
    """
    Compute 48-element summary vector for each simulated dataset.

    Parameters
    ----------
    simulations : np.ndarray
        Array of shape (n_sim, num_days, num_toads)

    Returns
    -------
    np.ndarray
        Array of shape (n_sim, 48)
    """
    n_sim = simulations.shape[0]
    summaries = np.zeros((n_sim, 48))
    for i in range(n_sim):
        try:
            summaries[i, :] = compute_displacement_summaries(simulations[i])
        except Exception:
            summaries[i, :] = np.nan
    return summaries

