import os
import re
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    Activation, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", required=True, choices=["C0", "C1", "C2"])
parser.add_argument("--target_len", type=int, default=150)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

CAMERA_TO_USE = args.camera
TARGET_LEN = args.target_len
EPOCHS = args.epochs

# =========================================================
# Reproducibility
# =========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# Paths
# =========================================================
DATASET_PATH = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results_tcn/{CAMERA_TO_USE}_tcn_len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Camera:", CAMERA_TO_USE)
print("Target length:", TARGET_LEN)
print("Output:", OUTPUT_DIR)

# =========================================================
# Load labels
# =========================================================
labels_df = pd.read_csv(CSV_PATH)

labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

# IMPORTANT:
# label depends on exercise/person/trial, not camera
score_map = {}
for _, row in labels_df.iterrows():
    key = (row["exercise"], row["person"], row["trial"])
    score_map[key] = float(row["mean"])

print("Total score entries:", len(score_map))

# =========================================================
# Load segment samples
# =========================================================
file_pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")
segment_samples = []

bad_files = []

for file_name in sorted(os.listdir(DATASET_PATH)):
    if not file_name.endswith(".npz"):
        continue

    match = file_pattern.search(file_name)
    if not match:
        continue

    exercise, person, trial, camera, seg_id = match.groups()

    if camera != CAMERA_TO_USE:
        continue

    score_key = (exercise, person, trial)
    if score_key not in score_map:
        continue

    file_path = os.path.join(DATASET_PATH, file_name)

    try:
        npz_data = np.load(file_path)
        keypoints_3d = npz_data["keypoints_3d"]
    except Exception as e:
        bad_files.append((file_name, str(e)))
        continue

    # FIX: skip corrupted / invalid segments
    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (17, 3):
        bad_files.append((file_name, keypoints_3d.shape))
        continue

    segment_samples.append({
        "file_name": file_name,
        "exercise": exercise,
        "person": person,
        "trial": trial,
        "camera": camera,
        "segment": int(seg_id),
        "x": keypoints_3d.astype(np.float32),
        "y": score_map[score_key]
    })

print("Valid raw segment samples:", len(segment_samples))
print("Bad / skipped files:", len(bad_files))

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w") as f:
    for item in bad_files:
        f.write(str(item) + "\n")

if len(segment_samples) == 0:
    raise ValueError("No valid segment samples loaded.")

# =========================================================
# Merge segments into full trials
# =========================================================
grouped_segments = defaultdict(list)

for sample in segment_samples:
    group_key = (
        sample["exercise"],
        sample["person"],
        sample["trial"],
        sample["camera"]
    )
    grouped_segments[group_key].append(
        (sample["segment"], sample["x"], sample["y"])
    )

trial_samples = []

for group_key, segs in grouped_segments.items():
    segs = sorted(segs, key=lambda item: item[0])

    valid_arrays = []

    for seg_id, arr, label in segs:
        if arr is None:
            continue

        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            print(f"Skipping bad segment inside merge: seg={seg_id}, shape={arr.shape}")
            continue

        valid_arrays.append(arr)

    if len(valid_arrays) == 0:
        print("Skipping trial because all segments are invalid:", group_key)
        continue

    full_sequence = np.concatenate(valid_arrays, axis=0)
    label = float(segs[0][2])

    exercise, person, trial, camera = group_key

    trial_samples.append({
        "exercise": exercise,
        "person": person,
        "trial": trial,
        "camera": camera,
        "x": full_sequence,
        "y": label
    })

print("Grouped trial samples:", len(trial_samples))

if len(trial_samples) == 0:
    raise ValueError("No grouped trials after filtering.")

# =========================================================
# Trial split
# =========================================================
unique_trials = sorted({
    (sample["exercise"], sample["person"], sample["trial"])
    for sample in trial_samples
})

train_trials, test_trials = train_test_split(
    unique_trials,
    test_size=0.2,
    random_state=SEED
)

train_trials_set = set(train_trials)

train_samples = []
test_samples = []

for sample in trial_samples:
    key = (sample["exercise"], sample["person"], sample["trial"])
    if key in train_trials_set:
        train_samples.append(sample)
    else:
        test_samples.append(sample)

print("Train samples:", len(train_samples))
print("Test samples :", len(test_samples))

pd.DataFrame(train_trials, columns=["exercise", "person", "trial"]).to_csv(
    os.path.join(OUTPUT_DIR, "train_trials.csv"),
    index=False
)

pd.DataFrame(test_trials, columns=["exercise", "person", "trial"]).to_csv(
    os.path.join(OUTPUT_DIR, "test_trials.csv"),
    index=False
)

# =========================================================
# Preprocessing
# =========================================================
def center_skeleton(seq_3d, root_idx=0):
    root = seq_3d[:, root_idx:root_idx + 1, :]
    return seq_3d - root


def smooth_sequence(seq_2d, window_size=5):
    pad = window_size // 2
    padded = np.pad(seq_2d, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.zeros_like(seq_2d, dtype=np.float32)

    for t in range(seq_2d.shape[0]):
        smoothed[t] = padded[t:t + window_size].mean(axis=0)

    return smoothed


def resample_sequence(seq_2d, target_len):
    old_len = seq_2d.shape[0]

    if old_len == target_len:
        return seq_2d.astype(np.float32)

    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, seq_2d.shape[1]), dtype=np.float32)

    for j in range(seq_2d.shape[1]):
        resampled[:, j] = np.interp(new_idx, old_idx, seq_2d[:, j])

    return resampled


def normalize_per_sample(seq_2d):
    mean = seq_2d.mean(axis=0, keepdims=True)
    std = seq_2d.std(axis=0, keepdims=True) + 1e-8
    return (seq_2d - mean) / std


def prepare_xy(sample_list, target_len):
    X = []
    y = []

    for sample in sample_list:
        seq_3d = sample["x"].copy()

        seq_3d = center_skeleton(seq_3d, root_idx=0)

        seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)  # (T, 51)

        seq_2d = smooth_sequence(seq_2d, window_size=5)
        seq_2d = resample_sequence(seq_2d, target_len)
        seq_2d = normalize_per_sample(seq_2d)

        X.append(seq_2d)
        y.append(float(sample["y"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_train, y_train = prepare_xy(train_samples, TARGET_LEN)
X_test, y_test = prepare_xy(test_samples, TARGET_LEN)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# =========================================================
# Scale target
# =========================================================
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# =========================================================
# TCN model
# =========================================================
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    shortcut = x

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same")(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x


input_layer = Input(shape=(TARGET_LEN, X_train.shape[2]))

x = tcn_block(input_layer, filters=64, kernel_size=3, dilation_rate=1)
x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2)
x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=4)
x = tcn_block(x, filters=128, kernel_size=3, dilation_rate=8)

x = GlobalAveragePooling1D()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)

output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=Huber(delta=1.0),
    metrics=["mae"]
)

model.summary()

# =========================================================
# Train
# =========================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train,
    y_train_scaled,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =========================================================
# Save training log
# =========================================================
with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i + 1}\n")
        f.write(f"Train Loss = {history.history['loss'][i]}\n")
        f.write(f"Val Loss   = {history.history['val_loss'][i]}\n")
        f.write(f"Train MAE  = {history.history['mae'][i]}\n")
        f.write(f"Val MAE    = {history.history['val_mae'][i]}\n\n")

# =========================================================
# Evaluate
# =========================================================
y_pred_scaled = model.predict(X_test).flatten()
test_loss, test_mae_scaled = model.evaluate(X_test, y_test_scaled, verbose=0)

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.clip(y_pred, 1.0, 5.0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== TCN Results =====")
print("Camera =", CAMERA_TO_USE)
print("Target_len =", TARGET_LEN)
print("MAE  =", mae)
print("RMSE =", rmse)
print("R2   =", r2)
print("Test Loss =", test_loss)
print("Test MAE scaled =", test_mae_scaled)

# =========================================================
# Save metrics
# =========================================================
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write("===== TCN Results =====\n")
    f.write(f"Camera = {CAMERA_TO_USE}\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"MAE  = {mae}\n")
    f.write(f"RMSE = {rmse}\n")
    f.write(f"R2   = {r2}\n")
    f.write(f"Test Loss = {test_loss}\n")
    f.write(f"Test MAE scaled = {test_mae_scaled}\n")
    f.write(f"Train samples = {len(train_samples)}\n")
    f.write(f"Test samples  = {len(test_samples)}\n")
    f.write(f"X_train shape = {X_train.shape}\n")
    f.write(f"X_test shape  = {X_test.shape}\n")
    f.write(f"Bad skipped files = {len(bad_files)}\n")

# =========================================================
# Save predictions
# =========================================================
results_df = pd.DataFrame({
    "True": y_test,
    "Predicted": y_pred,
    "Abs_Error": np.abs(y_test - y_pred)
})

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)

print(results_df.head(20))

# =========================================================
# Save plots
# =========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.axhline(y=test_loss, linestyle="--", label="Test Loss")
plt.title(f"TCN Loss ({CAMERA_TO_USE})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.axhline(y=test_mae_scaled, linestyle="--", label="Test MAE scaled")
plt.title(f"TCN MAE ({CAMERA_TO_USE})")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "training_curves.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([1, 5], [1, 5], "r--")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title(f"TCN True vs Predicted ({CAMERA_TO_USE})")
plt.savefig(
    os.path.join(OUTPUT_DIR, "true_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
