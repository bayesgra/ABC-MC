import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Load all sources
# ============================================================

# Load ABC distance-based results (.npz)
abc_results = np.load("normal_example_m0_distanceABC_results.npz", allow_pickle=True)

# Extract model probabilities (assuming shape [n_obs, n_dist, n_models])
model_probs = abc_results["model_probs"]  # shape: (100, n_dist, 3)
distance_names = list(abc_results["distance_names"])
model_labels = list(abc_results["model_labels"])

# We focus on Model 0 probabilities
model0_probs = model_probs[:, :, 0]  # shape: (100, n_dist)
df_abc = pd.DataFrame(model0_probs, columns=[f"ABC-{d}" for d in distance_names])

# Load other methods’ CSVs
df_nn = pd.read_csv("normal_model0_NN_probabilities.csv")
df_qda = pd.read_csv("normal_model0_QDA_probabilities.csv")
df_sa  = pd.read_csv("normal_model0_SA_probabilities.csv")

# Standardize column names and extract model 0 column
nn_col = df_nn.columns[0]
qda_col = df_qda.columns[0]
sa_col = df_sa.columns[0]

df_nn = df_nn.rename(columns={nn_col: "NN"})
df_qda = df_qda.rename(columns={qda_col: "ABC-QDA"})
df_sa = df_sa.rename(columns={sa_col: "ABC-SA"})

# ============================================================
# 2. Combine everything into a single DataFrame
# ============================================================
df_all = pd.concat([df_abc, df_sa, df_qda, df_nn], axis=1)

# Optional: rename columns for clarity/order
col_order = [c for c in df_abc.columns] + ["ABC-SA", "ABC-QDA", "NN"]
df_all = df_all[col_order]

# ============================================================
# 3. Save combined dataframe
# ============================================================
os.makedirs("results", exist_ok=True)
df_all.to_csv("results/normal_model0_combined_probabilities.csv", index=False)
print(" Combined probability data saved to results/normal_model0_combined_probabilities.csv")

# ============================================================
# 4. Boxplot visualization
# ============================================================
plt.figure(figsize=(10, 6))
df_all.boxplot()
plt.ylabel("Posterior probability for Model 0")
plt.title("Comparison of Model 0 posterior probabilities across ABC and NN methods")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("results/normal_model0_probabilities_boxplot.png", dpi=300)
plt.show()

