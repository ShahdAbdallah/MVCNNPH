
import os
import re
import random
import argparse
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--target_exercise", type=str, default="E1")
parser.add_argument("--target_len", type=int, default=120)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

TARGET_EXERCISE = args.target_exercise.strip()
TARGET_LEN = args.target_len
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DATASET = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_ROOT = os.path.join(BASE_DATASET, "by_person")
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

REQUIRED_CAMERAS = {"C0", "C1", "C2"}

OUTPUT_DIR = (
    f"/mvdlph/shahd/MVCNNPH/"
    f"results_multiview_{TARGET_EXERCISE}_early_fusion_velocity_only_len120_with_train_val_test/"
    f"len{TARGET_LEN}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Experiment: Multi-view Early Fusion Velocity Only")
print("Target exercise:", TARGET_EXERCISE)
print("Fusion: Early fusion / Concatenation")
print("Required cameras:", REQUIRED_CAMERAS)
print("Target length:", TARGET_LEN)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Output:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    score_map[(row["exercise"], row["person"], row["trial"])] = float(row["mean"])

pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")


def safe_pcc(y_true, y_pred):
    if len(y_true) <= 1:
        return np.nan, np.nan
    try:
        return pearsonr(y_true, y_pred)
    except Exception:
        return np.nan, np.nan


def compute_regression_metrics(y_true, y_pred):
    pcc, pcc_pvalue = safe_pcc(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "PCC": pcc,
        "PCC_pvalue": pcc_pvalue,
    }


# ============================================================
# Data loading + camera/label checks
# ============================================================
def load_multiview_split(split_name):
    split_dir = os.path.join(SPLIT_ROOT, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    segment_samples = []
    bad_files = []
    skipped_not_target = 0
    debug_rows = []

    for file_name in sorted(os.listdir(split_dir)):
        if not file_name.endswith(".npz"):
            continue

        match = pattern.search(file_name)
        if not match:
            bad_files.append((file_name, "filename pattern not matched"))
            continue

        exercise, person, trial, camera, seg_id = match.groups()

        if exercise != TARGET_EXERCISE:
            skipped_not_target += 1
            continue

        label_key = (exercise, person, trial)
        if label_key not in score_map:
            bad_files.append((file_name, "missing label"))
            continue

        file_path = os.path.join(split_dir, file_name)
        try:
            npz_data = np.load(file_path)
            keypoints_3d = npz_data["keypoints_3d"]
        except Exception as e:
            bad_files.append((file_name, str(e)))
            continue

        if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (17, 3) or keypoints_3d.shape[0] == 0:
            bad_files.append((file_name, f"bad shape {keypoints_3d.shape}"))
            continue

        label_value = score_map[label_key]

        if len(debug_rows) < 150:
            debug_rows.append({
                "split": split_name,
                "file_name": file_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "camera": camera,
                "segment": int(seg_id),
                "label_key": str(label_key),
                "label_mean": label_value,
                "keypoints_shape": str(keypoints_3d.shape),
            })

        segment_samples.append({
            "file_name": file_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "segment": int(seg_id),
            "x": keypoints_3d.astype(np.float32),
            "y": label_value,
        })

    camera_grouped = defaultdict(list)
    for sample in segment_samples:
        key = (sample["exercise"], sample["person"], sample["trial"], sample["camera"])
        camera_grouped[key].append((sample["segment"], sample["x"], sample["y"], sample["file_name"]))

    trial_camera_sequences = defaultdict(dict)
    for key, segs in camera_grouped.items():
        exercise, person, trial, camera = key
        segs = sorted(segs, key=lambda item: item[0])

        arrays = []
        for _, arr, _, _ in segs:
            if arr.ndim == 3 and arr.shape[1:] == (17, 3) and arr.shape[0] > 0:
                arrays.append(arr)
            else:
                bad_files.append((str(key), f"invalid segment shape {arr.shape}"))

        if not arrays:
            bad_files.append((str(key), "no valid arrays after filtering"))
            continue

        try:
            full_sequence = np.concatenate(arrays, axis=0)
        except Exception as e:
            bad_files.append((str(key), f"concatenate error: {e}"))
            continue

        trial_key = (exercise, person, trial)
        trial_camera_sequences[trial_key][camera] = {
            "x": full_sequence.astype(np.float32),
            "y": float(segs[0][2]),
            "file_name": segs[0][3],
            "num_segments": len(segs),
            "full_sequence_shape": str(full_sequence.shape),
        }

    multiview_samples = []
    incomplete_trials = []
    camera_alignment_rows = []
    label_mismatch_rows = []

    for trial_key, cam_dict in trial_camera_sequences.items():
        exercise, person, trial = trial_key
        available = set(cam_dict.keys())

        if not REQUIRED_CAMERAS.issubset(available):
            incomplete_trials.append({
                "split": split_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "available_cameras": ",".join(sorted(available)),
                "missing_cameras": ",".join(sorted(REQUIRED_CAMERAS - available)),
            })
            continue

        c0_label = cam_dict["C0"]["y"]
        c1_label = cam_dict["C1"]["y"]
        c2_label = cam_dict["C2"]["y"]

        if not (c0_label == c1_label == c2_label):
            label_mismatch_rows.append({
                "split": split_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "c0_label": c0_label,
                "c1_label": c1_label,
                "c2_label": c2_label,
            })

        camera_alignment_rows.append({
            "split": split_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "label": c0_label,
            "c0_label": c0_label,
            "c1_label": c1_label,
            "c2_label": c2_label,
            "c0_file": cam_dict["C0"]["file_name"],
            "c1_file": cam_dict["C1"]["file_name"],
            "c2_file": cam_dict["C2"]["file_name"],
            "c0_frames": cam_dict["C0"]["x"].shape[0],
            "c1_frames": cam_dict["C1"]["x"].shape[0],
            "c2_frames": cam_dict["C2"]["x"].shape[0],
            "c0_shape": cam_dict["C0"]["full_sequence_shape"],
            "c1_shape": cam_dict["C1"]["full_sequence_shape"],
            "c2_shape": cam_dict["C2"]["full_sequence_shape"],
            "c0_segments": cam_dict["C0"]["num_segments"],
            "c1_segments": cam_dict["C1"]["num_segments"],
            "c2_segments": cam_dict["C2"]["num_segments"],
        })

        multiview_samples.append({
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "x_c0": cam_dict["C0"]["x"],
            "x_c1": cam_dict["C1"]["x"],
            "x_c2": cam_dict["C2"]["x"],
            "y": c0_label,
        })

    return multiview_samples, bad_files, incomplete_trials, skipped_not_target, debug_rows, camera_alignment_rows, label_mismatch_rows


def samples_to_df(samples, split_name):
    rows = []
    for s in samples:
        rows.append({
            "split": split_name,
            "exercise": s["exercise"],
            "person": s["person"],
            "trial": s["trial"],
            "score": s["y"],
            "performance": "good" if s["y"] >= 4.0 else "bad",
        })
    return pd.DataFrame(rows)


# ============================================================
# Preprocessing
# ============================================================
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


def add_velocity_features(seq_2d):
    velocity = np.zeros_like(seq_2d, dtype=np.float32)
    velocity[1:] = seq_2d[1:] - seq_2d[:-1]
    return np.concatenate([seq_2d, velocity], axis=1)


def preprocess_one_view(seq_3d):
    seq_3d = center_skeleton(seq_3d)
    seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)
    seq_2d = smooth_sequence(seq_2d, window_size=5)
    seq_2d = resample_sequence(seq_2d, TARGET_LEN)
    seq_2d = normalize_per_sample(seq_2d)
    seq_2d = add_velocity_features(seq_2d)
    return seq_2d.astype(np.float32)


def prepare_xy(samples):
    X, y, meta = [], [], []

    for sample in samples:
        c0 = preprocess_one_view(sample["x_c0"])
        c1 = preprocess_one_view(sample["x_c1"])
        c2 = preprocess_one_view(sample["x_c2"])

        fused = np.concatenate([c0, c1, c2], axis=1)  # (TARGET_LEN, 306)

        X.append(fused)
        y.append(float(sample["y"]))
        meta.append({
            "exercise": sample["exercise"],
            "person": sample["person"],
            "trial": sample["trial"],
        })

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


# ============================================================
# Load data
# ============================================================
fit_samples, bad_train, incomplete_train, skipped_train, debug_train, alignment_train, mismatch_train = load_multiview_split("train")
val_samples, bad_valid, incomplete_valid, skipped_valid, debug_valid, alignment_valid, mismatch_valid = load_multiview_split("valid")
test_samples, bad_test, incomplete_test, skipped_test, debug_test, alignment_test, mismatch_test = load_multiview_split("test")

bad_files = bad_train + bad_valid + bad_test
incomplete_trials = incomplete_train + incomplete_valid + incomplete_test
debug_rows = debug_train + debug_valid + debug_test
camera_alignment_rows = alignment_train + alignment_valid + alignment_test
label_mismatch_rows = mismatch_train + mismatch_valid + mismatch_test

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w", encoding="utf-8") as f:
    for item in bad_files:
        f.write(str(item) + "\n")

pd.DataFrame(incomplete_trials).to_csv(os.path.join(OUTPUT_DIR, "incomplete_camera_trials.csv"), index=False)
pd.DataFrame(debug_rows).to_csv(os.path.join(OUTPUT_DIR, "label_camera_debug_first_files.csv"), index=False)
pd.DataFrame(camera_alignment_rows).to_csv(os.path.join(OUTPUT_DIR, "camera_alignment_check.csv"), index=False)
pd.DataFrame(label_mismatch_rows).to_csv(os.path.join(OUTPUT_DIR, "label_mismatch_check.csv"), index=False)

print("Fit samples:", len(fit_samples))
print("Validation samples:", len(val_samples))
print("Test samples:", len(test_samples))
print("Bad files:", len(bad_files))
print("Incomplete camera trials:", len(incomplete_trials))
print("Label mismatches:", len(label_mismatch_rows))
print("Skipped non-target files:", skipped_train + skipped_valid + skipped_test)

if len(fit_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
    raise ValueError("One split has zero samples. Check complete-camera availability and target exercise.")

fit_dist_df = samples_to_df(fit_samples, "train")
val_dist_df = samples_to_df(val_samples, "valid")
test_dist_df = samples_to_df(test_samples, "test")

split_distribution_df = pd.concat([fit_dist_df, val_dist_df, test_dist_df], ignore_index=True)
split_distribution_df.to_csv(os.path.join(OUTPUT_DIR, "split_distribution.csv"), index=False)

person_summary_df = split_distribution_df.groupby(["split", "person"]).agg(
    num_samples=("score", "count"),
    mean_score=("score", "mean"),
    min_score=("score", "min"),
    max_score=("score", "max"),
    good_count=("performance", lambda x: (x == "good").sum()),
    bad_count=("performance", lambda x: (x == "bad").sum()),
).reset_index()
person_summary_df.to_csv(os.path.join(OUTPUT_DIR, "person_split_summary.csv"), index=False)

exercise_summary_df = split_distribution_df.groupby(["split", "exercise"]).agg(
    num_samples=("score", "count"),
    mean_score=("score", "mean"),
    min_score=("score", "min"),
    max_score=("score", "max"),
    good_count=("performance", lambda x: (x == "good").sum()),
    bad_count=("performance", lambda x: (x == "bad").sum()),
).reset_index()
exercise_summary_df.to_csv(os.path.join(OUTPUT_DIR, "exercise_split_summary.csv"), index=False)

X_fit, y_fit, fit_meta = prepare_xy(fit_samples)
X_val, y_val, val_meta = prepare_xy(val_samples)
X_test, y_test, test_meta = prepare_xy(test_samples)

print("X_fit shape:", X_fit.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

y_scaler = StandardScaler()
y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

feature_dim = X_fit.shape[2]

# ============================================================
# Model
# ============================================================
seq_input = Input(shape=(TARGET_LEN, feature_dim), name="early_fusion_input")

x = Conv1D(16, 5, activation="relu", padding="same")(seq_input)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(24, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(24, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)

x = Conv1D(32, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling1D()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.10)(x)
output = Dense(1)(x)

model = Model(inputs=seq_input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)
model.summary()


class RegressionMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train_scaled, X_val, y_val_scaled):
        super().__init__()
        self.X_train = X_train
        self.y_train_scaled = y_train_scaled
        self.X_val = X_val
        self.y_val_scaled = y_val_scaled
        self.train_r2 = []
        self.val_r2 = []
        self.train_pcc = []
        self.val_pcc = []

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(self.X_train, verbose=0).flatten()
        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        self.train_r2.append(r2_score(self.y_train_scaled, train_pred))
        self.val_r2.append(r2_score(self.y_val_scaled, val_pred))
        train_pcc, _ = safe_pcc(self.y_train_scaled, train_pred)
        val_pcc, _ = safe_pcc(self.y_val_scaled, val_pred)
        self.train_pcc.append(train_pcc)
        self.val_pcc.append(val_pcc)


epoch_metrics = RegressionMetricsCallback(X_fit, y_fit_scaled, X_val, y_val_scaled)
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")

callbacks = [
    epoch_metrics,
    EarlyStopping(monitor="val_loss", patience=80, min_delta=0.00005, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
]

history = model.fit(
    X_fit,
    y_fit_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i + 1}\n")
        f.write(f"Train Loss = {history.history['loss'][i]}\n")
        f.write(f"Val Loss = {history.history['val_loss'][i]}\n")
        f.write(f"Train MAE scaled = {history.history['mae'][i]}\n")
        f.write(f"Val MAE scaled = {history.history['val_mae'][i]}\n")
        f.write(f"Train RMSE scaled = {history.history['rmse'][i]}\n")
        f.write(f"Val RMSE scaled = {history.history['val_rmse'][i]}\n")
        f.write(f"Train R2 scaled = {epoch_metrics.train_r2[i]}\n")
        f.write(f"Val R2 scaled = {epoch_metrics.val_r2[i]}\n")
        f.write(f"Train PCC scaled = {epoch_metrics.train_pcc[i]}\n")
        f.write(f"Val PCC scaled = {epoch_metrics.val_pcc[i]}\n\n")

# ============================================================
# Train / validation / test evaluation
# ============================================================
train_pred_scaled = model.predict(X_fit, verbose=0).flatten()
val_pred_scaled = model.predict(X_val, verbose=0).flatten()
test_pred_scaled = model.predict(X_test, verbose=0).flatten()

train_loss, train_mae_scaled, train_rmse_scaled = model.evaluate(X_fit, y_fit_scaled, verbose=0)
val_loss, val_mae_scaled, val_rmse_scaled = model.evaluate(X_val, y_val_scaled, verbose=0)
test_loss, test_mae_scaled, test_rmse_scaled = model.evaluate(X_test, y_test_scaled, verbose=0)

train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

train_pred = np.clip(train_pred, 1.0, 5.0)
val_pred = np.clip(val_pred, 1.0, 5.0)
test_pred = np.clip(test_pred, 1.0, 5.0)

train_metrics = compute_regression_metrics(y_fit, train_pred)
val_metrics = compute_regression_metrics(y_val, val_pred)
test_metrics = compute_regression_metrics(y_test, test_pred)

print("\n===== TRAIN RESULTS =====")
print(train_metrics)
print("Loss scaled =", train_loss, "MAE scaled =", train_mae_scaled, "RMSE scaled =", train_rmse_scaled)

print("\n===== VALIDATION RESULTS =====")
print(val_metrics)
print("Loss scaled =", val_loss, "MAE scaled =", val_mae_scaled, "RMSE scaled =", val_rmse_scaled)

print("\n===== TEST RESULTS =====")
print(test_metrics)
print("Loss scaled =", test_loss, "MAE scaled =", test_mae_scaled, "RMSE scaled =", test_rmse_scaled)

split_metrics_df = pd.DataFrame([
    {"split": "train", "loss_scaled": train_loss, "mae_scaled": train_mae_scaled, "rmse_scaled": train_rmse_scaled, **train_metrics},
    {"split": "validation", "loss_scaled": val_loss, "mae_scaled": val_mae_scaled, "rmse_scaled": val_rmse_scaled, **val_metrics},
    {"split": "test", "loss_scaled": test_loss, "mae_scaled": test_mae_scaled, "rmse_scaled": test_rmse_scaled, **test_metrics},
])
split_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "train_validation_test_metrics.csv"), index=False)


def create_prediction_df(meta_list, y_true, y_pred, split_name):
    rows = []
    for meta, true_value, pred_value in zip(meta_list, y_true, y_pred):
        rows.append({
            "split": split_name,
            "exercise": meta["exercise"],
            "person": meta["person"],
            "trial": meta["trial"],
            "true_score": true_value,
            "predicted_score": pred_value,
            "abs_error": abs(true_value - pred_value),
        })
    return pd.DataFrame(rows)


train_predictions_df = create_prediction_df(fit_meta, y_fit, train_pred, "train")
val_predictions_df = create_prediction_df(val_meta, y_val, val_pred, "validation")
test_predictions_df = create_prediction_df(test_meta, y_test, test_pred, "test")
all_predictions_df = pd.concat([train_predictions_df, val_predictions_df, test_predictions_df], ignore_index=True)

train_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)
val_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "validation_predictions.csv"), index=False)
test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
all_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "all_split_predictions.csv"), index=False)

worst_predictions_df = test_predictions_df.sort_values("abs_error", ascending=False).head(30)
worst_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "worst_30_predictions.csv"), index=False)

per_person_rows = []
for person, group in test_predictions_df.groupby("person"):
    true = group["true_score"].values
    pred = group["predicted_score"].values
    person_pcc, _ = safe_pcc(true, pred)
    per_person_rows.append({
        "person": person,
        "num_test_trials": len(group),
        "mean_true_score": np.mean(true),
        "mean_predicted_score": np.mean(pred),
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred) if len(group) > 1 else np.nan,
        "PCC": person_pcc,
    })

per_person_df = pd.DataFrame(per_person_rows).sort_values("person")
per_person_df.to_csv(os.path.join(OUTPUT_DIR, "per_person_scores.csv"), index=False)

# ============================================================
# metrics.txt
# ============================================================
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"===== Multi-view {TARGET_EXERCISE} Early Fusion Velocity Only Results =====\n")
    f.write(f"Target exercise = {TARGET_EXERCISE}\n")
    f.write("Fusion type = Early fusion\n")
    f.write("Fusion method = Concatenate C0, C1, C2 along feature dimension before CNN\n")
    f.write("Views = C0 + C1 + C2\n")
    f.write("Complete-camera rule = if any camera is missing, the full trial is skipped\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"Epochs requested = {EPOCHS}\n")
    f.write(f"Epochs trained = {len(history.history['loss'])}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write("Loss = MSE\n")
    f.write("Learning rate = 2e-5\n")
    f.write("Filters = 16, 24, 24, 32\n")
    f.write("Dense layers = 32\n")
    f.write("Dropout = 0.10\n")
    f.write("Added features = velocity only\n")
    f.write("Input per view = 102\n")
    f.write("Fused input features = 306\n")
    f.write("Exercise embedding = No\n")
    f.write("Oversampling = No\n")
    f.write("Augmentation = No\n")
    f.write("Extra checks = camera_alignment_check.csv, label_mismatch_check.csv, train/validation/test metrics\n")
    f.write("Images archive = images.tar.gz\n\n")

    f.write("===== Train Metrics =====\n")
    f.write(f"Train MAE = {train_metrics['MAE']}\n")
    f.write(f"Train RMSE = {train_metrics['RMSE']}\n")
    f.write(f"Train R2 = {train_metrics['R2']}\n")
    f.write(f"Train PCC = {train_metrics['PCC']}\n")
    f.write(f"Train PCC p-value = {train_metrics['PCC_pvalue']}\n")
    f.write(f"Train Loss scaled = {train_loss}\n")
    f.write(f"Train MAE scaled = {train_mae_scaled}\n")
    f.write(f"Train RMSE scaled = {train_rmse_scaled}\n\n")

    f.write("===== Validation Metrics =====\n")
    f.write(f"Validation MAE = {val_metrics['MAE']}\n")
    f.write(f"Validation RMSE = {val_metrics['RMSE']}\n")
    f.write(f"Validation R2 = {val_metrics['R2']}\n")
    f.write(f"Validation PCC = {val_metrics['PCC']}\n")
    f.write(f"Validation PCC p-value = {val_metrics['PCC_pvalue']}\n")
    f.write(f"Validation Loss scaled = {val_loss}\n")
    f.write(f"Validation MAE scaled = {val_mae_scaled}\n")
    f.write(f"Validation RMSE scaled = {val_rmse_scaled}\n\n")

    f.write("===== Test Metrics =====\n")
    f.write(f"Test MAE = {test_metrics['MAE']}\n")
    f.write(f"Test RMSE = {test_metrics['RMSE']}\n")
    f.write(f"Test R2 = {test_metrics['R2']}\n")
    f.write(f"Test PCC = {test_metrics['PCC']}\n")
    f.write(f"Test PCC p-value = {test_metrics['PCC_pvalue']}\n")
    f.write(f"Test Loss scaled = {test_loss}\n")
    f.write(f"Test MAE scaled = {test_mae_scaled}\n")
    f.write(f"Test RMSE scaled = {test_rmse_scaled}\n\n")

    f.write("===== Dataset Sizes =====\n")
    f.write(f"Fit samples = {len(fit_samples)}\n")
    f.write(f"Validation samples = {len(val_samples)}\n")
    f.write(f"Test samples = {len(test_samples)}\n")
    f.write(f"X_fit shape = {X_fit.shape}\n")
    f.write(f"X_val shape = {X_val.shape}\n")
    f.write(f"X_test shape = {X_test.shape}\n")
    f.write(f"Bad skipped files = {len(bad_files)}\n")
    f.write(f"Incomplete camera trials = {len(incomplete_trials)}\n")
    f.write(f"Label mismatches = {len(label_mismatch_rows)}\n")
    f.write(f"Skipped non-target files = {skipped_train + skipped_valid + skipped_test}\n\n")

    f.write("===== Split Person Summary =====\n")
    f.write(person_summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Split Exercise Summary =====\n")
    f.write(exercise_summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Per-person test scores =====\n")
    f.write(per_person_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Worst 30 test predictions =====\n")
    f.write(worst_predictions_df.to_string(index=False))

np.savez(
    os.path.join(OUTPUT_DIR, "plot_data.npz"),
    train_loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"]),
    train_mae=np.array(history.history["mae"]),
    val_mae=np.array(history.history["val_mae"]),
    train_rmse=np.array(history.history["rmse"]),
    val_rmse=np.array(history.history["val_rmse"]),
    train_r2=np.array(epoch_metrics.train_r2),
    val_r2=np.array(epoch_metrics.val_r2),
    train_pcc=np.array(epoch_metrics.train_pcc),
    val_pcc=np.array(epoch_metrics.val_pcc),
    final_train_loss=np.array([train_loss]),
    final_val_loss=np.array([val_loss]),
    test_loss=np.array([test_loss]),
    train_mae_original=np.array([train_metrics["MAE"]]),
    val_mae_original=np.array([val_metrics["MAE"]]),
    test_mae_original=np.array([test_metrics["MAE"]]),
    train_rmse_original=np.array([train_metrics["RMSE"]]),
    val_rmse_original=np.array([val_metrics["RMSE"]]),
    test_rmse_original=np.array([test_metrics["RMSE"]]),
    train_r2_original=np.array([train_metrics["R2"]]),
    val_r2_original=np.array([val_metrics["R2"]]),
    test_r2_original=np.array([test_metrics["R2"]]),
    y_train=y_fit,
    y_train_pred=train_pred,
    y_val=y_val,
    y_val_pred=val_pred,
    y_test=y_test,
    y_pred=test_pred,
)

# ============================================================
# Plots
# ============================================================
def get_ylim(values, pad_ratio=0.15):
    clean_values = [float(v) for v in values if not pd.isna(v)]
    vmin = min(clean_values)
    vmax = max(clean_values)
    pad = (vmax - vmin) * pad_ratio if vmax != vmin else 0.1
    return vmin - pad, vmax + pad


loss_values = history.history["loss"]
val_loss_values = history.history["val_loss"]
all_loss = loss_values + val_loss_values + [test_loss]
loss_ymin, loss_ymax = get_ylim(all_loss)

plt.figure(figsize=(9, 5))
plt.plot(loss_values, label="Train", linewidth=2)
plt.plot(val_loss_values, label="Validation", linewidth=2)
plt.axhline(y=test_loss, linestyle="-.", linewidth=2, label=f"Test Loss={test_loss:.4f}")
plt.title(f"{TARGET_EXERCISE} Early Fusion Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(loss_ymin, loss_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

all_rmse = history.history["rmse"] + history.history["val_rmse"] + [test_rmse_scaled]
rmse_ymin, rmse_ymax = get_ylim(all_rmse)

all_mae = history.history["mae"] + history.history["val_mae"] + [test_mae_scaled]
mae_ymin, mae_ymax = get_ylim(all_mae)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["rmse"], label="Train", linewidth=2)
plt.plot(history.history["val_rmse"], label="Validation", linewidth=2)
plt.axhline(y=test_rmse_scaled, linestyle="-.", linewidth=2, label=f"Test RMSE={test_rmse_scaled:.4f}")
plt.title("RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.ylim(rmse_ymin, rmse_ymax)
plt.grid(True, alpha=0.25)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train", linewidth=2)
plt.plot(history.history["val_mae"], label="Validation", linewidth=2)
plt.axhline(y=test_mae_scaled, linestyle="-.", linewidth=2, label=f"Test MAE={test_mae_scaled:.4f}")
plt.title("MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.ylim(mae_ymin, mae_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.suptitle(f"{TARGET_EXERCISE} Early Fusion RMSE & MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_mae.png"), dpi=300, bbox_inches="tight")
plt.close()

all_r2 = epoch_metrics.train_r2 + epoch_metrics.val_r2 + [test_metrics["R2"], 0, 1]
r2_ymin, r2_ymax = get_ylim(all_r2, pad_ratio=0.10)

plt.figure(figsize=(9, 5))
plt.plot(epoch_metrics.train_r2, label="Train", linewidth=2)
plt.plot(epoch_metrics.val_r2, label="Validation", linewidth=2)
plt.axhline(y=test_metrics["R2"], linestyle="-.", linewidth=2, label=f"Test R2={test_metrics['R2']:.4f}")
plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="Baseline")
plt.title(f"{TARGET_EXERCISE} Early Fusion R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.ylim(r2_ymin, r2_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "r2_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

all_pcc = epoch_metrics.train_pcc + epoch_metrics.val_pcc + [test_metrics["PCC"], 0, 1]
pcc_ymin, pcc_ymax = get_ylim(all_pcc, pad_ratio=0.10)

plt.figure(figsize=(9, 5))
plt.plot(epoch_metrics.train_pcc, label="Train", linewidth=2)
plt.plot(epoch_metrics.val_pcc, label="Validation", linewidth=2)
plt.axhline(y=test_metrics["PCC"], linestyle="-.", linewidth=2, label=f"Test PCC={test_metrics['PCC']:.4f}")
plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="No correlation")
plt.title(f"{TARGET_EXERCISE} Early Fusion PCC")
plt.xlabel("Epoch")
plt.ylabel("PCC")
plt.ylim(pcc_ymin, pcc_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pcc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, alpha=0.7)
plt.plot([1, 5], [1, 5], "r--")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title(f"{TARGET_EXERCISE} Early Fusion True vs Predicted")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

image_files = [
    "loss_curve.png",
    "rmse_mae.png",
    "r2_curve.png",
    "pcc_curve.png",
    "true_vs_predicted.png",
]

with tarfile.open(os.path.join(OUTPUT_DIR, "images.tar.gz"), "w:gz") as tar:
    for img in image_files:
        img_path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(img_path):
            tar.add(img_path, arcname=img)

print("\nSaved files in:", OUTPUT_DIR)
print("metrics.txt")
print("train_validation_test_metrics.csv")
print("training_log.txt")
print("train_predictions.csv")
print("validation_predictions.csv")
print("predictions.csv")
print("all_split_predictions.csv")
print("worst_30_predictions.csv")
print("per_person_scores.csv")
print("split_distribution.csv")
print("person_split_summary.csv")
print("exercise_split_summary.csv")
print("camera_alignment_check.csv")
print("label_camera_debug_first_files.csv")
print("label_mismatch_check.csv")
print("plot_data.npz")
print("loss_curve.png")
print("rmse_mae.png")
print("r2_curve.png")
print("pcc_curve.png")
print("true_vs_predicted.png")
print("images.tar.gz")
print("bad_files.txt")
print("incomplete_camera_trials.csv")
print("best_model.keras")
