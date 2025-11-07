# toad_example_NN.py
import os, sys, random, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from enum import Enum
from scipy.stats import levy_stable

# If you need to import shared utils from the project root, keep this line:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ========================
# Reproducibility
# ========================
def set_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed set to {seed}")

# ========================
# Model enum
# ========================
class Model(Enum):
    RANDOM = 0
    NEAREST = 1
    DISTANCE = 2

# ========================
# Toad movement simulation
# ========================
def distance_based_probs(position: float, refuge_locations: np.ndarray, p0: float, d0: float) -> np.ndarray:
    """
    Probability of returning to each previous refuge, decaying with distance.
    """
    if refuge_locations.size == 0:
        # If no refuges yet, return a dummy prob (will be normalized by caller)
        return np.array([1.0])
    refuge_distances = np.abs(position - refuge_locations)
    return p0 * np.exp(-refuge_distances / d0)

def toad_movement_sample(
    model: Model,
    alpha: float,
    gamma: float,
    p0: float,
    d0: float = None,
    num_toads: int = 66,
    num_days: int = 63
) -> np.ndarray:
    """
    Simulate toad positions over time for a chosen return model.
    Returns an array of shape (num_days, num_toads).
    """
    # positions
    toad_positions = np.zeros((num_days, num_toads))

    # Model-specific state
    if model == Model.DISTANCE:
        refuge_counts = np.ones(num_toads, dtype=int)
        refuge_locations = np.zeros((num_days, num_toads))  # [time, toad]
    else:
        # For RANDOM / NEAREST, use constant no-return probability
        no_return_probs = 1 - p0

    # Levy-stable steps
    # alpha \in (0,2]; here alpha in ~[1,2), beta=0 (sym), scale=gamma
    steps = levy_stable.rvs(alpha, 0, scale=gamma, size=(num_days - 1, num_toads))

    for i in range(1, num_days):
        new_pos = toad_positions[i - 1] + steps[i - 1]

        if model == Model.DISTANCE:
            # Compute per-toad vector of return probabilities to each refuge
            # then convert to no-return probs by product over (1 - prob)
            refuge_probs = [
                distance_based_probs(
                    new_pos[j],
                    refuge_locations[:refuge_counts[j], j],
                    p0,
                    d0
                )
                for j in range(num_toads)
            ]
            no_return_probs_vec = np.array([
                np.prod(1.0 - (rp / np.clip(np.sum(rp), 1e-12, None)))  # normalize to avoid >1
                for rp in refuge_probs
            ])
        else:
            no_return_probs_vec = np.full(num_toads, no_return_probs)

        # Decide return vs no-return
        no_return_flag = (np.random.uniform(size=num_toads) < no_return_probs_vec)
        no_return_ids = np.nonzero(no_return_flag)[0]
        return_ids = np.nonzero(~no_return_flag)[0]

        # Non-returners move to new_pos
        toad_positions[i, no_return_ids] = new_pos[no_return_ids]

        # Returners choose a previous location
        if model == Model.RANDOM:
            # Uniformly choose a previous day
            if i > 0 and return_ids.size > 0:
                return_location_ids = np.random.randint(0, i, size=return_ids.shape)
                toad_positions[i, return_ids] = toad_positions[return_location_ids, return_ids]
        elif model == Model.NEAREST:
            # Choose nearest previous location
            if i > 0 and return_ids.size > 0:
                prev_positions = toad_positions[:i, return_ids]  # [time, toads]
                diffs = np.abs(new_pos[return_ids] - prev_positions)  # [time, toads]
                idx = np.argmin(diffs, axis=0)
                toad_positions[i, return_ids] = prev_positions[idx, np.arange(return_ids.size)]
        else:
            # DISTANCE: choose a previous refuge with distance-based probabilities
            if return_ids.size > 0:
                for ridx, j in enumerate(return_ids):
                    rp = refuge_probs[j]
                    rp = rp / np.clip(np.sum(rp), 1e-12, None)  # normalize
                    choose_idx = np.random.choice(np.arange(refuge_counts[j]), p=rp)
                    toad_positions[i, j] = refuge_locations[choose_idx, j]

            # Update refuge set for non-returners
            if no_return_ids.size > 0:
                refuge_locations[refuge_counts[no_return_ids], no_return_ids] = new_pos[no_return_ids]
                refuge_counts[no_return_ids] += 1

    return toad_positions

# ========================
# Dataset simulation
# ========================
def simulate_datasets(
    num_samples: int,
    num_toads: int = 66,
    num_days: int = 63
):
    """
    Simulate training data evenly across the three models using fixed 'true' parameters
    (as per your earlier convention), no normalization.
    Returns:
      X: (N, num_days, num_toads)
      y_class: one-hot (N, 3)
      y_params: (N, 4) with (alpha, gamma, p0, d0). For non-DISTANCE models, d0 = 0.
    """
    # Fixed params (feel free to adjust to your canonical set)
    # RANDOM
    params_random  = (1.70, 34.0, 0.60, 0.0)    # (alpha, gamma, p0, d0=0)
    # NEAREST
    params_nearest = (1.83, 46.0, 0.65, 0.0)
    # DISTANCE
    params_distance = (1.65, 32.0, 0.43, 758.0)

    X = []
    y_model = []
    y_params = []

    per_model = num_samples // 3
    triples = [
        (Model.RANDOM,  params_random),
        (Model.NEAREST, params_nearest),
        (Model.DISTANCE, params_distance),
    ]

    for label, (m, pars) in enumerate(triples):
        for _ in range(per_model):
            sim = toad_movement_sample(
                model=m,
                alpha=pars[0], gamma=pars[1], p0=pars[2], d0=pars[3],
                num_toads=num_toads, num_days=num_days
            )
            X.append(sim)
            y_model.append(label)
            y_params.append(pars)

    X = np.stack(X)                               # (N, num_days, num_toads)
    y_model = to_categorical(np.array(y_model), num_classes=3)
    y_params = np.array(y_params, dtype=np.float32)  # (N, 4)

    print(f"Simulated {num_samples:,} datasets with shape {X.shape}")
    return X, y_model, y_params

# ========================
# NN model
# ========================
def build_model(num_days: int, num_toads: int):
    """
    Multi-task model: classification (3 classes) + parameter regression (alpha, gamma, p0, d0).
    """
    inp = Input(shape=(num_days, num_toads))
    x = layers.Flatten()(inp)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    out_class = layers.Dense(3, activation='softmax', name='class_output')(x)
    out_param = layers.Dense(4, activation='linear', name='param_output')(x)

    model = models.Model(inputs=inp, outputs=[out_class, out_param])
    model.compile(
        optimizer='adam',
        loss={'class_output': 'categorical_crossentropy', 'param_output': 'mse'},
        loss_weights={'class_output': 1.0, 'param_output': 0.5},
        metrics={'class_output': 'accuracy', 'param_output': 'mae'}
    )
    return model

# ========================
# Train
# ========================
def train_model(model, X, y_class, y_params, output_dir, epochs=10, batch_size=32, random_state=42):
    X_train, X_test, y_class_train, y_class_test, y_params_train, y_params_test = train_test_split(
        X, y_class, y_params, test_size=0.2, random_state=random_state
    )
    print("Training model...")
    model.fit(
        X_train,
        {'class_output': y_class_train, 'param_output': y_params_train},
        epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1
    )
    res = model.evaluate(
        X_test, {'class_output': y_class_test, 'param_output': y_params_test}, verbose=0
    )
    # Keras metric ordering: [total_loss, class_loss, param_loss, class_acc, param_mae]
    print(f"Test Classification Accuracy: {res[3]:.4f} | Test Param MAE: {res[4]:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'toad_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

# ========================
# Predict on observed
# ========================
def predict_observed(model_path, observed_path, output_dir):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading observed datasets from {observed_path}")
    data = np.load(observed_path, allow_pickle=True)
    X_new = data['observed_datasets']  # expected shape: (n_obs, num_days, num_toads)

    pred_class_probs, pred_params = model.predict(X_new, verbose=1)

    os.makedirs(output_dir, exist_ok=True)
    # Probabilities CSV
    prob_df = pd.DataFrame(
        pred_class_probs,
        columns=['NN_Prob_RANDOM', 'NN_Prob_NEAREST', 'NN_Prob_DISTANCE']
    )
    prob_df.to_csv(os.path.join(output_dir, 'toad_NN_probabilities.csv'), index=False)

    # Parameters CSV (alpha, gamma, p0, d0)
    param_df = pd.DataFrame(
        pred_params,
        columns=['NN_alpha', 'NN_gamma', 'NN_p0', 'NN_d0']
    )
    param_df.to_csv(os.path.join(output_dir, 'toad_NN_params.csv'), index=False)

    print(f"Predictions saved in {output_dir}")

# ========================
# Main
# ========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the NN on simulated data')
    parser.add_argument('--predict', action='store_true', help='Predict on observed_datasets.npz')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=120000)  # 40k per model by default
    parser.add_argument('--num_toads', type=int, default=66)
    parser.add_argument('--num_days', type=int, default=63)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_seed(args.seed)

    model_dir = os.path.join(RESULTS_DIR, 'NN')
    os.makedirs(model_dir, exist_ok=True)
    observed_path = os.path.join(DATA_DIR, 'observed_datasets.npz')

    model_path = os.path.join(model_dir, 'toad_model.h5')

    if args.train:
        X, y_class, y_params = simulate_datasets(
            num_samples=args.num_samples,
            num_toads=args.num_toads,
            num_days=args.num_days
        )
        model = build_model(args.num_days, args.num_toads)
        model_path = train_model(
            model, X, y_class, y_params, model_dir,
            epochs=args.epochs, batch_size=args.batch_size
        )

    if args.predict:
        if not os.path.exists(observed_path):
            raise FileNotFoundError(
                f"Observed data not found at {observed_path}. "
                f"Make sure you have saved data/observed_datasets.npz from your generator."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run with --train first."
            )
        predict_observed(model_path, observed_path, model_dir)

if __name__ == '__main__':
    main()

