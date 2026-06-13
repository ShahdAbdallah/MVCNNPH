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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


parser = argparse.ArgumentParser()
parser.add_argument("--target_len", type=int, default=150)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

TARGET_LEN = args.target_len
EPOCHS = args.epochs

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results_allviews_stratified_cnn/allviews_len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Mode: All camera views as single-view samples")
print("Target length:", TARGET_LEN)
print("Epochs:", EPOCHS)
print("Output directory:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    key = (row["exercise"], row["person"], row["trial"])
    score_map[key] = float(row["mean"])

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

    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (17, 3) or keypoints_3d.shape[0] == 0:
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

print("Valid segments:", len(segment_samples))
print("Bad / skipped files:", len(bad_files))

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w") as f:
    for item in bad_files:
        f.write(str(item) + "\n")

if len(segment_samples) == 0:
    raise ValueError("No valid segment samples loaded.")

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

view_samples = []

for group_key, segs in grouped_segments.items():
    segs = sorted(segs, key=lambda item: item[0])

    valid_arrays = []

    for seg_id, arr, label in segs:
        if arr.ndim == 3 and arr.shape[1:] == (17, 3) and arr.shape[0] > 0:
            valid_arrays.append(arr)

    if len(valid_arrays) == 0:
        continue

    full_sequence = np.concatenate(valid_arrays, axis=0)
    label = float(segs[0][2])

    exercise, person, trial, camera = group_key

    view_samples.append({
        "exercise": exercise,
        "person": person,
        "trial": trial,
        "camera": camera,
        "x": full_sequence,
        "y": label
    })

print("Grouped single-view samples:", len(view_samples))

if len(view_samples) == 0:
    raise ValueError("No grouped view samples after filtering.")

trial_df = pd.DataFrame([
    {
        "exercise": s["exercise"],
        "person": s["person"],
        "trial": s["trial"],
        "score": s["y"]
    }
    for s in view_samples
]).drop_duplicates(subset=["exercise", "person", "trial"])

trial_df["performance"] = trial_df["score"].apply(lambda x: "good" if x >= 4.0 else "bad")
trial_df["stratify_key"] = trial_df["exercise"] + "_" + trial_df["performance"]

print("\nStratification distribution:")
print(trial_df["stratify_key"].value_counts())

strata_counts = trial_df["stratify_key"].value_counts()
too_small = strata_counts[strata_counts < 2]

if len(too_small) > 0:
    print("\nWarning: Some exercise-performance strata have less than 2 samples.")
    print(too_small)
    print("Falling back to stratification by exercise only.")

    stratify_col = trial_df["exercise"]

    if (stratify_col.value_counts() < 2).any():
        print("Some exercises have less than 2 samples. Falling back to random split.")
        stratify_col = None
else:
    stratify_col = trial_df["stratify_key"]

train_df, test_df = train_test_split(
    trial_df,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_col
)

train_keys = set(zip(train_df["exercise"], train_df["person"], train_df["trial"]))
test_keys = set(zip(test_df["exercise"], test_df["person"], test_df["trial"]))

train_samples = []
test_samples = []

for sample in view_samples:
    key = (sample["exercise"], sample["person"], sample["trial"])

    if key in train_keys:
        train_samples.append(sample)
    elif key in test_keys:
        test_samples.append(sample)

print("\nTrain single-view samples:", len(train_samples))
print("Test single-view samples :", len(test_samples))

train_df.to_csv(os.path.join(OUTPUT_DIR, "train_trials.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_trials.csv"), index=False)

split_summary = {
    "train_trials": len(train_df),
    "test_trials": len(test_df),
    "train_view_samples": len(train_samples),
    "test_view_samples": len(test_samples)
}

with open(os.path.join(OUTPUT_DIR, "split_summary.txt"), "w") as f:
    f.write("Split summary\n")
    for k, v in split_summary.items():
        f.write(f"{k}: {v}\n")
    f.write("\nTrain performance distribution:\n")
    f.write(str(train_df["performance"].value_counts()))
    f.write("\n\nTest performance distribution:\n")
    f.write(str(test_df["performance"].value_counts()))
    f.write("\n\nTrain exercise distribution:\n")
    f.write(str(train_df["exercise"].value_counts().sort_index()))
    f.write("\n\nTest exercise distribution:\n")
    f.write(str(test_df["exercise"].value_counts().sort_index()))


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
    meta = []

    for sample in sample_list:
        seq_3d = sample["x"].copy()

        seq_3d = center_skeleton(seq_3d, root_idx=0)
        seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)

        seq_2d = smooth_sequence(seq_2d, window_size=5)
        seq_2d = resample_sequence(seq_2d, target_len)
        seq_2d = normalize_per_sample(seq_2d)

        X.append(seq_2d)
        y.append(float(sample["y"]))

        meta.append({
            "exercise": sample["exercise"],
            "person": sample["person"],
            "trial": sample["trial"],
            "camera": sample["camera"]
        })

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


X_train, y_train, train_meta = prepare_xy(train_samples, TARGET_LEN)
X_test, y_test, test_meta = prepare_xy(test_samples, TARGET_LEN)

print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

feature_dim = X_train.shape[2]

model = Sequential([
    Conv1D(32, 5, activation="relu", input_shape=(TARGET_LEN, feature_dim)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 3, activation="relu"),
    GlobalAveragePooling1D(),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=Huber(delta=1.0),
    metrics=["mae"]
)

model.summary()

history = model.fit(
    X_train,
    y_train_scaled,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=8,
    verbose=1
)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i + 1}\n")
        f.write(f"Train Loss = {history.history['loss'][i]}\n")
        f.write(f"Val Loss   = {history.history['val_loss'][i]}\n")
        f.write(f"Train MAE  = {history.history['mae'][i]}\n")
        f.write(f"Val MAE    = {history.history['val_mae'][i]}\n\n")

y_pred_scaled = model.predict(X_test).flatten()
test_loss, test_mae_scaled = model.evaluate(X_test, y_test_scaled, verbose=0)

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.clip(y_pred, 1.0, 5.0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== All-Views Single-View CNN Results =====")
print("MAE  =", mae)
print("RMSE =", rmse)
print("R2   =", r2)
print("Test Loss =", test_loss)
print("Test MAE scaled =", test_mae_scaled)

pred_rows = []

for meta, true_value, pred_value in zip(test_meta, y_test, y_pred):
    pred_rows.append({
        "exercise": meta["exercise"],
        "person": meta["person"],
        "trial": meta["trial"],
        "camera": meta["camera"],
        "true_score": true_value,
        "predicted_score": pred_value,
        "abs_error": abs(true_value - pred_value)
    })

predictions_df = pd.DataFrame(pred_rows)
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

per_exercise_rows = []

for exercise, group in predictions_df.groupby("exercise"):
    ex_true = group["true_score"].values
    ex_pred = group["predicted_score"].values

    ex_mae = mean_absolute_error(ex_true, ex_pred)
    ex_rmse = np.sqrt(mean_squared_error(ex_true, ex_pred))

    if len(group) >= 2:
        ex_r2 = r2_score(ex_true, ex_pred)
    else:
        ex_r2 = np.nan

    per_exercise_rows.append({
        "exercise": exercise,
        "num_test_samples": len(group),
        "mean_true_score": np.mean(ex_true),
        "mean_predicted_score": np.mean(ex_pred),
        "MAE": ex_mae,
        "RMSE": ex_rmse,
        "R2": ex_r2
    })

per_exercise_df = pd.DataFrame(per_exercise_rows).sort_values("exercise")
per_exercise_df.to_csv(os.path.join(OUTPUT_DIR, "per_exercise_scores.csv"), index=False)

per_camera_rows = []

for camera, group in predictions_df.groupby("camera"):
    cam_true = group["true_score"].values
    cam_pred = group["predicted_score"].values

    cam_mae = mean_absolute_error(cam_true, cam_pred)
    cam_rmse = np.sqrt(mean_squared_error(cam_true, cam_pred))

    if len(group) >= 2:
        cam_r2 = r2_score(cam_true, cam_pred)
    else:
        cam_r2 = np.nan

    per_camera_rows.append({
        "camera": camera,
        "num_test_samples": len(group),
        "mean_true_score": np.mean(cam_true),
        "mean_predicted_score": np.mean(cam_pred),
        "MAE": cam_mae,
        "RMSE": cam_rmse,
        "R2": cam_r2
    })

per_camera_df = pd.DataFrame(per_camera_rows).sort_values("camera")
per_camera_df.to_csv(os.path.join(OUTPUT_DIR, "per_camera_scores.csv"), index=False)

print("\nPer-exercise scores:")
print(per_exercise_df.to_string(index=False))

print("\nPer-camera scores:")
print(per_camera_df.to_string(index=False))

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write("===== All-Views Single-View CNN Results =====\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"Epochs = {EPOCHS}\n")
    f.write(f"MAE  = {mae}\n")
    f.write(f"RMSE = {rmse}\n")
    f.write(f"R2   = {r2}\n")
    f.write(f"Test Loss = {test_loss}\n")
    f.write(f"Test MAE scaled = {test_mae_scaled}\n")
    f.write(f"Train single-view samples = {len(train_samples)}\n")
    f.write(f"Test single-view samples  = {len(test_samples)}\n")
    f.write(f"Train unique trials = {len(train_df)}\n")
    f.write(f"Test unique trials  = {len(test_df)}\n")
    f.write(f"X_train shape = {X_train.shape}\n")
    f.write(f"X_test shape  = {X_test.shape}\n")
    f.write(f"Bad skipped files = {len(bad_files)}\n\n")

    f.write("Train performance distribution:\n")
    f.write(str(train_df["performance"].value_counts()))
    f.write("\n\nTest performance distribution:\n")
    f.write(str(test_df["performance"].value_counts()))
    f.write("\n\nTrain exercise distribution:\n")
    f.write(str(train_df["exercise"].value_counts().sort_index()))
    f.write("\n\nTest exercise distribution:\n")
    f.write(str(test_df["exercise"].value_counts().sort_index()))
    f.write("\n\nPer-exercise scores:\n")
    f.write(per_exercise_df.to_string(index=False))
    f.write("\n\nPer-camera scores:\n")
    f.write(per_camera_df.to_string(index=False))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.axhline(y=test_loss, linestyle="--", label="Test Loss")
plt.title("All-Views Single-View CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.axhline(y=test_mae_scaled, linestyle="--", label="Test MAE scaled")
plt.title("All-Views Single-View CNN MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([1, 5], [1, 5], "r--")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("All-Views Single-View CNN True vs Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved files:")
print("metrics.txt")
print("predictions.csv")
print("per_exercise_scores.csv")
print("per_camera_scores.csv")
print("split_summary.txt")
print("training_log.txt")
print("training_curves.png")
print("true_vs_predicted.png")
