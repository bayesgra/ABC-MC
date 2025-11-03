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

def compute_displacement_summaries(positions):
    """
    Compute simple summary statistics from simulated toad trajectories.
    """
    displacements = np.diff(positions, axis=1)
    mean_disp = np.mean(np.abs(displacements), axis=1)
    var_disp = np.var(displacements, axis=1)
    return np.vstack((mean_disp, var_disp)).T


def summarise_sample(position_sample):
    """
    Summarize the trajectory sample by mean and variance of displacements.
    """
    summaries = compute_displacement_summaries(position_sample)
    return np.mean(summaries, axis=0)


def get_statistics(simulations):
    """
    Convert multiple trajectory samples into a matrix of summary statistics.
    """
    return np.array([summarise_sample(sim) for sim in simulations])
