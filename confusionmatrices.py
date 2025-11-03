import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Load probabilities
# ============================================================
df = pd.read_csv("results/normal_model0_combined_probabilities.csv")

# ============================================================
# 2. Random predictions based on probabilities
#    (for each entry, sample model ∈ {0,1} with P(model=0)=prob)
# ============================================================
rng = np.random.default_rng(seed=42)
predictions = pd.DataFrame(index=df.index, columns=df.columns)

for col in df.columns:
    probs = df[col].values
    preds = rng.binomial(n=1, p=probs)  # 1 = predict Model 0
    predictions[col] = preds

# ============================================================
# 3. Compute absolute counts of Model 0 predictions
# ============================================================
model0_counts = predictions.sum(axis=0).rename("Count_Model0")

# ============================================================
# 4. Save and print results
# ============================================================
print("Absolute frequencies of Model 0 predictions per method:")
print(model0_counts)

model0_counts.to_csv("results/normal_model0_random_prediction_counts.csv", header=True)
print("\n Saved to results/normal_model0_random_prediction_counts.csv")

# ============================================================
# 5. Confusion matrices
# ============================================================
# Define the confusion matrices (rows: actual, cols: predicted)
methods = {
    "ABC-CvM": [
        [100, 0, 0, 0, 0, 0],
        [98, 2, 0, 0, 0, 0],
        [85, 0, 15, 0, 0, 0],
        [60, 0, 0, 40, 0, 0],
        [18, 0, 0, 0, 82, 0],
        [3, 0, 0, 0, 0, 97],
    ],
    "ABC-MMD": [
        [100, 0, 0, 0, 0, 0],
        [98, 2, 0, 0, 0, 0],
        [85, 0, 15, 0, 0, 0],
        [61, 0, 0, 39, 0, 0],
        [18, 0, 0, 0, 82, 0],
        [4, 0, 0, 0, 0, 96],
    ],
    "ABC-Wass": [
        [100, 0, 0, 0, 0, 0],
        [98, 2, 0, 0, 0, 0],
        [86, 0, 14, 0, 0, 0],
        [59, 0, 0, 41, 0, 0],
        [17, 0, 0, 0, 83, 0],
        [2, 0, 0, 0, 0, 98],
    ],
    "ABC-Stat": [
        [99, 1, 0, 0, 0, 0],
        [97, 3, 0, 0, 0, 0],
        [81, 0, 19, 0, 0, 0],
        [52, 0, 0, 48, 0, 0],
        [15, 0, 0, 0, 85, 0],
        [2, 0, 0, 0, 0, 98],
    ],
    "NN": [
        [50, 50, 0, 0, 0, 0],
        [45, 55, 0, 0, 0, 0],
        [45, 0, 55, 0, 0, 0],
        [47, 0, 0, 53, 0, 0],
        [48, 0, 0, 0, 52, 0],
        [40, 0, 0, 0, 0, 60],
    ],
    "ABC-SA": [
        [97, 3, 0, 0, 0, 0],
        [95, 5, 0, 0, 0, 0],
        [84, 0, 16, 0, 0, 0],
        [62, 0, 0, 38, 0, 0],
        [45, 0, 0, 0, 55, 0],
        [28, 0, 0, 0, 0, 72],
    ],
    "ABC-QDA": [
        [100, 0, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
        [100, 0, 0, 0, 0, 0],
    ]
}

# Compute global vmin and vmax across all matrices
all_values = []
for matrix in methods.values():
    arr = np.array(matrix).astype(float)
    arr[arr == 0] = np.nan  # We'll ignore zeros for the scale
    all_values.extend(arr[~np.isnan(arr)].flatten())

global_vmin = min(all_values)
global_vmax = max(all_values)

# 2. Generate heatmaps with fixed vmin and vmax
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
axes = axes.flatten()

for i, (method, matrix) in enumerate(methods.items()):
    arr = np.array(matrix).astype(float)
    arr[arr == 0] = np.nan  # Replace zeros with NaN
    df = pd.DataFrame(arr, index=[f"{i/10:.1f}" for i in range(6)],
                            columns=[f"{i/10:.1f}" for i in range(6)])
    
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(df, annot=True, fmt=".0f", cmap=cmap,
                vmin=global_vmin, vmax=global_vmax,
                center=50, cbar=True, linewidths=.5, square=True,
                mask=df.isna(), ax=axes[i])
    axes[i].set_title(method)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Hide the last subplot if there are fewer than 8 methods
if len(methods) < len(axes):
    for j in range(len(methods), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("normal_example_CM.png", dpi=300, bbox_inches='tight')
plt.show()
