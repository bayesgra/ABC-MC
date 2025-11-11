import os, sys, random, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ========================
# Helpers
# ========================
def set_seed(seed=12345):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    print(f"Seed set to {seed}")

def simulate_datasets(num_datasets=10**6, sample_size=100):
    n_per_dist = num_datasets // 2
    normal_samples_m0 = np.random.normal(0, 1, size=(n_per_dist, sample_size))
    normal_samples_m1 = np.random.normal(0.1, 1, size=(n_per_dist, sample_size))
    X = np.vstack([normal_samples_m0, normal_samples_m1])
    #X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    y_class = np.vstack([np.zeros((n_per_dist, 1)), np.ones((n_per_dist, 1))])
    y_class = to_categorical(y_class, num_classes=2)
    y_params = np.vstack([np.zeros((n_per_dist, 1)), np.full((n_per_dist, 1), 0.1)])
    print(f"Simulated {num_datasets:,} datasets (sample size {sample_size})")
    return X, y_class, y_params

def build_model(sample_size):
    inp = Input(shape=(sample_size,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    out_class = layers.Dense(2, activation='softmax', name='class_output')(x)
    out_param = layers.Dense(1, activation='linear', name='param_output')(x)
    model = models.Model(inputs=inp, outputs=[out_class, out_param])
    model.compile(optimizer='adam', loss={'class_output': 'categorical_crossentropy', 'param_output': 'mse'}, loss_weights={'class_output': 1.0, 'param_output': 0.5}, metrics={'class_output': 'accuracy', 'param_output': 'mae'})
    return model

def train_model(model, X, y_class, y_params, output_dir, epochs=10, batch_size=32):
    X_train, X_test, y_class_train, y_class_test, y_params_train, y_params_test = train_test_split(X, y_class, y_params, test_size=0.2, random_state=42)
    print("Training model...")
    model.fit(X_train, {'class_output': y_class_train, 'param_output': y_params_train}, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    res = model.evaluate(X_test, {'class_output': y_class_test, 'param_output': y_params_test}, verbose=0)
    print(f"Test Accuracy: {res[3]:.4f} | MAE: {res[4]:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'normal_example_m0_NN_model.keras'))
    print(f"Model saved in {output_dir}/normal_example_m0_NN_model.keras")
    return model

def predict_observed(model_path, observed_path, output_dir):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loading observed datasets from {observed_path}")
    X_new = np.load(observed_path)
    #X_new = (X_new - np.mean(X_new, axis=1, keepdims=True)) / np.std(X_new, axis=1, keepdims=True)
    pred_class_probs, pred_params = model.predict(X_new, verbose=1)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pred_class_probs[:, 0], columns=['NN_Prob_Model0']).to_csv(os.path.join(output_dir, 'normal_model0_NN_probabilities.csv'), index=False)
    pd.DataFrame(pred_params[:, 0], columns=['NN_Param_Estimate']).to_csv(os.path.join(output_dir, 'normal_model0_NN_params.csv'), index=False)
    print(f"Predictions saved in {output_dir}")

# ========================
# Main
# ========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_seed(args.seed)

    model_output = os.path.join(RESULTS_DIR, 'NN')
    os.makedirs(model_output, exist_ok=True)
    observed_path = os.path.join(DATA_DIR, 'observed_datasets.npy')

    if args.train:
        X, y_class, y_params = simulate_datasets(sample_size=args.sample_size)
        model = build_model(args.sample_size)
        train_model(model, X, y_class, y_params, model_output, args.epochs, args.batch_size)

    if args.predict:
        model_path = os.path.join(model_output, 'normal_example_m0_NN_model.keras')
        if not os.path.exists(observed_path):
            raise FileNotFoundError(f"Observed data not found at {observed_path}. Run distanceABC first.")
        predict_observed(model_path, observed_path, model_output)

if __name__ == '__main__':
    main()
