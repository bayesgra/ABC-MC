import os, sys, csv, numpy as np, pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import toad_utils

# ================================================================
# Import shared utilities
# ================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import abc_utils


# ================================================================
# QDA simulation for one observed dataset
# ================================================================
def simulate_for_observed(obs_data, obs_id, n_simulations):
    rng = np.random.default_rng(obs_id)
    obs_labels = np.zeros(len(obs_data))

    def simulate_once(seed):
        rng = np.random.default_rng(seed)
        dist_type = rng.integers(0, 3)  # 0=RANDOM, 1=NEAREST, 2=DISTANCE

        # Sample parameters (simulate variability)
        alpha = rng.uniform(1.0, 2.0)
        gamma = rng.uniform(0.0, 1.0)
        p0 = rng.uniform(0.2, 0.5)
        d0 = rng.uniform(0.5, 2.0) if dist_type == 2 else None

        # Generate simulated data
        if dist_type == 0:
            sim_data = toad_movement_sample("random", alpha, gamma, p0)
        elif dist_type == 1:
            sim_data = toad_movement_sample("nearest", alpha, gamma, p0)
        else:
            sim_data = toad_movement_sample("distance", alpha, gamma, p0, d0=d0)

        # Train QDA classifier
        X = np.concatenate([obs_data, sim_data]).reshape(-1, 1)
        y = np.concatenate([obs_labels, np.ones(len(sim_data))])
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        acc = accuracy_score(y, qda.predict(X))

        params = [alpha, gamma, p0] if dist_type in [0, 1] else [alpha, gamma, p0, d0]
        return acc, dist_type, params

    results = Parallel(n_jobs=-1)(delayed(simulate_once)(i) for i in range(n_simulations))

    accuracies = np.array([r[0] for r in results])
    dist_types = np.array([r[1] for r in results])
    threshold = np.percentile(accuracies, 10)
    idx = np.where(accuracies <= threshold)[0]

    selected = [results[i] for i in idx]
    print(f"Obs {obs_id:03d}: selected {len(selected)} / {n_simulations}")

    # Each row: obs_id, accuracy, model_id, params...
    output = []
    for acc, m, params in selected:
        row = [obs_id, acc, m] + params
        output.append(row)
    return output


# ================================================================
# Summarize QDA results into probabilities + parameter means
# ================================================================
def summarize_results(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    summary_rows = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        if df.empty:
            continue

        models = df["dist_type"].values
        probs = [np.mean(models == i) for i in [0, 1, 2]]

        # Compute parameter means per model
        mean_params = {}
        for model_id in [0, 1, 2]:
            subset = df[df["dist_type"] == model_id]
            if subset.empty:
                mean_params[model_id] = [np.nan] * (4 if model_id == 2 else 3)
            else:
                param_cols = [c for c in df.columns if c not in ["obs_id", "accuracy", "dist_type"]]
                max_len = 4 if model_id == 2 else 3
                arr = subset[param_cols].to_numpy(dtype=float)
                if arr.shape[1] < max_len:
                    arr = np.pad(arr, ((0, 0), (0, max_len - arr.shape[1])), constant_values=np.nan)
                mean_params[model_id] = np.nanmean(arr, axis=0).tolist()

        summary_rows.append({
            "ObsID": filename.split("_")[-1].split(".")[0],
            "Prob_Model0": probs[0],
            "Prob_Model1": probs[1],
            "Prob_Model2": probs[2],
            "Mean_Alpha_Model0": mean_params[0][0],
            "Mean_Gamma_Model0": mean_params[0][1],
            "Mean_P0_Model0": mean_params[0][2],
            "Mean_Alpha_Model1": mean_params[1][0],
            "Mean_Gamma_Model1": mean_params[1][1],
            "Mean_P0_Model1": mean_params[1][2],
            "Mean_Alpha_Model2": mean_params[2][0],
            "Mean_Gamma_Model2": mean_params[2][1],
            "Mean_P0_Model2": mean_params[2][2],
            "Mean_D0_Model2": mean_params[2][3],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(folder_path, "toad_QDA_summary.csv"), index=False)
    print(f" Saved full probability and parameter summary to {folder_path}/toad_QDA_summary.csv")

    # Save simple CSVs for backwards compatibility
    df_summary[["ObsID", "Prob_Model0", "Prob_Model1", "Prob_Model2"]].to_csv(
        os.path.join(folder_path, "toad_example_random_QDA_probabilities.csv"), index=False)
    param_cols = [c for c in df_summary.columns if c.startswith("Mean_")]
    df_summary[["ObsID"] + param_cols].to_csv(
        os.path.join(folder_path, "toad_example_random_QDA_params.csv"), index=False)


# ================================================================
# Main routine
# ================================================================
def main():
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "qdaABC")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    observed_path = os.path.join(DATA_DIR, "observed_datasets.npz")
    if not os.path.exists(observed_path):
        raise FileNotFoundError(f"Missing observed datasets at {observed_path}. Run toad_example_distanceABC.py first.")

    observed = np.load(observed_path, allow_pickle=True)["observed_datasets"]

    for obs_id, obs_data in enumerate(observed):
        results = simulate_for_observed(obs_data, obs_id, n_simulations=100000)
        filename = os.path.join(RESULTS_DIR, f"qda_simulations_obs_{obs_id:03d}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["obs_id", "accuracy", "dist_type", "alpha", "gamma", "p0", "d0"])
            writer.writerows(results)

    summarize_results(RESULTS_DIR)
    print(f" All QDA results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
