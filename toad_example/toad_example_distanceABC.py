# ============================
# toad_example_distanceABC.py
# ============================
import os, sys, numpy as np
from multiprocessing import Pool, cpu_count
import toad_utils

# Import shared utils from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc_utils import toad_movement_sample, distance_based_probs




def generate_observed_datasets(n_observed=100, num_toads=66, num_days=63, seed=42):
np.random.seed(seed)
observed = [toad_movement_sample('random', alpha=1.5, gamma=0.2, p0=0.3) for _ in range(n_observed)]
return np.array(observed, dtype=object)




def simulate_datasets(n_sim=10**6, num_toads=66, num_days=63):
alphas = np.random.uniform(1, 2, n_sim)
gammas = np.random.uniform(0, 1, n_sim)
models = np.random.choice(['random', 'distance'], size=n_sim)
sims = [toad_movement_sample(m, alpha=a, gamma=g, p0=0.3) for m, a, g in zip(models, alphas, gammas)]
return sims, alphas, gammas, models




def run_abc_for_one_observed(args):
obs, sims, alphas, gammas, models = args
distances = [np.mean(np.abs(obs - s)) for s in sims]
cutoff = np.percentile(distances, 1)
idx = np.where(distances <= cutoff)[0]
return {
'model_probs': {m: np.mean(np.array(models)[idx] == m) for m in np.unique(models)},
'theta_means': {
'alpha': np.mean(np.array(alphas)[idx]),
'gamma': np.mean(np.array(gammas)[idx])
}
}

def main():
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


observed = generate_observed_datasets()
np.savez(os.path.join(DATA_DIR, 'observed_datasets.npz'), observed_datasets=observed)
print(f" Saved observed datasets to {DATA_DIR}")


sims, alphas, gammas, models = simulate_datasets()
args_list = [(obs, sims, alphas, gammas, models) for obs in observed]


print(f"Running ABC on {len(observed)} datasets using {cpu_count()} cores...")
with Pool() as pool:
results = pool.map(run_abc_for_one_observed, args_list)


np.savez(os.path.join(RESULTS_DIR, 'toad_example_random_distanceABC_results.npz'), results=results)
print(f" Saved results to {RESULTS_DIR}")


if __name__ == '__main__':
main()