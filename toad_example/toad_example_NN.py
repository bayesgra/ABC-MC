# ================================================
# Load the observed datasets and predict
# ================================================
import os
import pandas as pd
import toad_utils

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "NN")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

observed_file = os.path.join(DATA_DIR, "observed_datasets.npz")
if not os.path.exists(observed_file):
    raise FileNotFoundError(f" Missing observed datasets at {observed_file}. Run toad_example_distanceABC.py first.")

data = np.load(observed_file, allow_pickle=True)
X_new = data["observed_datasets"]  # shape (n_observed, num_days, num_toads)
print(f" Loaded observed datasets: {X_new.shape}")

# ================================================
# Use the trained model to make predictions
# ================================================
pred_class_probs, pred_params = model.predict(X_new)

# Get most probable class for each observation
class_predictions = np.argmax(pred_class_probs, axis=1)

# ================================================
# Save raw prediction arrays (.npy)
# ================================================
np.save(os.path.join(RESULTS_DIR, "toad_class_predictions.npy"), class_predictions)
np.save(os.path.join(RESULTS_DIR, "toad_param_predictions.npy"), pred_params)
print(f" Saved raw predictions in {RESULTS_DIR}")

# ================================================
# Create CSV summaries
# ================================================
# Model names mapping
model_names = {0: "RANDOM", 1: "NEAREST", 2: "DISTANCE"}

# Build DataFrame with class probabilities and parameter estimates
df_results = pd.DataFrame({
    "ObsID": np.arange(len(X_new)),
    "Predicted_Model_ID": class_predictions,
    "Predicted_Model_Name": [model_names[m] for m in class_predictions],
    "Prob_Random": pred_class_probs[:, 0],
    "Prob_Nearest": pred_class_probs[:, 1],
    "Prob_Distance": pred_class_probs[:, 2],
    "Alpha": pred_params[:, 0],
    "Gamma": pred_params[:, 1],
    "P0": pred_params[:, 2],
    "D0": pred_params[:, 3],
})

# Mask irrelevant parameters (since RANDOM and NEAREST models have only 3 parameters)
for idx, model_id in enumerate(class_predictions):
    if model_id in [0, 1]:  # RANDOM or NEAREST
        df_results.loc[idx, "D0"] = np.nan  # remove extra param

# Save as CSV
csv_path = os.path.join(RESULTS_DIR, "toad_example_random_NN_params.csv")
df_results.to_csv(csv_path, index=False)
print(f" Saved detailed CSV results to {csv_path}")

# Also save a compact probabilities-only CSV
probs_csv = os.path.join(RESULTS_DIR, "toad_example_random_NN_probabilities.csv")
df_results[["ObsID", "Prob_Random", "Prob_Nearest", "Prob_Distance"]].to_csv(probs_csv, index=False)
print(f" Saved model probabilities to {probs_csv}")

# ================================================
# Save the trained model
# ================================================
model_path = os.path.join(RESULTS_DIR, "toad_example_random_NN_model.h5")
model.save(model_path)
print(f" Saved trained neural network model to {model_path}")

# ================================================
# Summary message
# ================================================
print("\n Prediction Summary:")
print(df_results.groupby("Predicted_Model_Name").size())
print(f"\nAll results saved in {RESULTS_DIR}")
