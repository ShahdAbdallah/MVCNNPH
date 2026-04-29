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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


parser = argparse.ArgumentParser()
parser.add_argument("--camera", required=True, choices=["C0", "C1", "C2"])
parser.add_argument("--mode", required=True, choices=["exp2", "exp3"])
parser.add_argument("--target_len", type=int, required=True)
args = parser.parse_args()

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

CAMERA_TO_USE = args.camera
MODE = args.mode
TARGET_LEN = args.target_len

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results/{CAMERA_TO_USE}_{MODE}_len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Camera:", CAMERA_TO_USE)
print("Mode:", MODE)
print("Target length:", TARGET_LEN)
print("Output:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()
labels_df["camera"] = labels_df["camera"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    key = (row["exercise"], row["person"], row["trial"])
    score_map[key] = float(row["mean"])

print("Total score entries:", len(score_map))

file_pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")
segment_samples = []

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
        print("Skipping", file_name, "error:", e)
        continue

    if keypoints_3d.ndim != 3:
        continue

    segment_samples.append({
        "file_name": file_name,
        "exercise": exercise,
        "person": person,
        "trial": trial,
        "camera": camera,
        "segment": int(seg_id),
        "x": keypoints_3d,
        "y": score_map[score_key]
    })

print("Raw segment samples:", len(segment_samples))

grouped_segments = defaultdict(list)

for sample in segment_samples:
    group_key = (sample["exercise"], sample["person"], sample["trial"], sample["camera"])
    grouped_segments[group_key].append((sample["segment"], sample["x"], sample["y"]))

trial_samples = []

for group_key, seg_list in grouped_segments.items():
    seg_list = sorted(seg_list, key=lambda item: item[0])
    full_sequence = np.concatenate([item[1] for item in seg_list], axis=0)
    label = float(seg_list[0][2])

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
    raise ValueError("No samples loaded. Check dataset path, labels, and camera matching.")

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
    os.path.join(OUTPUT_DIR, "train_trials.csv"), index=False
)
pd.DataFrame(test_trials, columns=["exercise", "person", "trial"]).to_csv(
    os.path.join(OUTPUT_DIR, "test_trials.csv"), index=False
)


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


def extract_few_features(seq_3d):
    joint_range = np.linalg.norm(seq_3d.max(axis=0) - seq_3d.min(axis=0), axis=1)

    if seq_3d.shape[0] > 1:
        vel = np.diff(seq_3d, axis=0)
        joint_vel = np.linalg.norm(vel, axis=2).mean(axis=0)
    else:
        joint_vel = np.zeros(17, dtype=np.float32)

    return np.concatenate([joint_range, joint_vel]).astype(np.float32)


def prepare_xy_exp2(sample_list, target_len):
    X = []
    y = []

    for sample in sample_list:
        seq_3d = center_skeleton(sample["x"].copy(), root_idx=0)
        seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)
        seq_2d = smooth_sequence(seq_2d, window_size=5)
        seq_2d = resample_sequence(seq_2d, target_len)
        seq_2d = normalize_per_sample(seq_2d)

        X.append(seq_2d)
        y.append(float(sample["y"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_xy_exp3(sample_list, target_len):
    X_seq = []
    X_feat = []
    y = []

    for sample in sample_list:
        seq_3d = center_skeleton(sample["x"].copy(), root_idx=0)

        feat = extract_few_features(seq_3d)

        seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)
        seq_2d = resample_sequence(seq_2d, target_len)
        seq_2d = normalize_per_sample(seq_2d)

        X_seq.append(seq_2d)
        X_feat.append(feat)
        y.append(float(sample["y"]))

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_feat, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


if MODE == "exp2":
    X_train, y_train = prepare_xy_exp2(train_samples, TARGET_LEN)
    X_test, y_test = prepare_xy_exp2(test_samples, TARGET_LEN)

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

elif MODE == "exp3":
    X_train_seq, X_train_feat, y_train = prepare_xy_exp3(train_samples, TARGET_LEN)
    X_test_seq, X_test_feat, y_test = prepare_xy_exp3(test_samples, TARGET_LEN)

    print("X_train_seq shape :", X_train_seq.shape)
    print("X_test_seq shape  :", X_test_seq.shape)
    print("X_train_feat shape:", X_train_feat.shape)
    print("X_test_feat shape :", X_test_feat.shape)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

if MODE == "exp3":
    feat_scaler = StandardScaler()
    X_train_feat = feat_scaler.fit_transform(X_train_feat)
    X_test_feat = feat_scaler.transform(X_test_feat)

if MODE == "exp2":
    feature_dim = X_train.shape[2]

    model = Sequential([
        Input(shape=(TARGET_LEN, feature_dim)),
        Conv1D(32, 5, activation="relu"),
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

else:
    seq_input = Input(shape=(TARGET_LEN, X_train_seq.shape[2]), name="seq_input")
    x = Conv1D(32, 5, activation="relu")(seq_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 3, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)

    feat_input = Input(shape=(X_train_feat.shape[1],), name="feat_input")
    f = Dense(32, activation="relu")(feat_input)

    merged = Concatenate()([x, f])
    merged = Dense(64, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(1)(merged)

    model = Model(inputs=[seq_input, feat_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=Huber(delta=1.0),
    metrics=["mae"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

if MODE == "exp2":
    history = model.fit(
        X_train,
        y_train_scaled,
        validation_split=0.2,
        epochs=1000,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
else:
    history = model.fit(
        [X_train_seq, X_train_feat],
        y_train_scaled,
        validation_split=0.2,
        epochs=1000,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i+1}:\n")
        f.write(f"Train Loss: {history.history['loss'][i]}\n")
        f.write(f"Val Loss  : {history.history['val_loss'][i]}\n")
        f.write(f"Train MAE : {history.history['mae'][i]}\n")
        f.write(f"Val MAE   : {history.history['val_mae'][i]}\n\n")

if MODE == "exp2":
    y_pred_scaled = model.predict(X_test).flatten()
    test_loss, test_mae_scaled = model.evaluate(X_test, y_test_scaled, verbose=0)
else:
    y_pred_scaled = model.predict([X_test_seq, X_test_feat]).flatten()
    test_loss, test_mae_scaled = model.evaluate([X_test_seq, X_test_feat], y_test_scaled, verbose=0)

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.clip(y_pred, 1.0, 5.0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== Results =====")
print("MAE  =", mae)
print("RMSE =", rmse)
print("R2   =", r2)
print("Test Loss =", test_loss)
print("Test MAE scaled =", test_mae_scaled)

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write("===== Results =====\n")
    f.write(f"Camera = {CAMERA_TO_USE}\n")
    f.write(f"Mode = {MODE}\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"MAE  = {mae}\n")
    f.write(f"RMSE = {rmse}\n")
    f.write(f"R2   = {r2}\n")
    f.write(f"Test Loss = {test_loss}\n")
    f.write(f"Test MAE scaled = {test_mae_scaled}\n")
    f.write(f"Train samples = {len(train_samples)}\n")
    f.write(f"Test samples  = {len(test_samples)}\n")

results_df = pd.DataFrame({
    "True": y_test,
    "Predicted": y_pred,
    "Abs_Error": np.abs(y_test - y_pred)
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.axhline(y=test_loss, color="r", linestyle="--", label="Test Loss")
plt.title(f"Loss {CAMERA_TO_USE} {MODE}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.axhline(y=test_mae_scaled, color="r", linestyle="--", label="Test MAE scaled")
plt.title(f"MAE {CAMERA_TO_USE} {MODE}")
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
plt.title(f"True vs Predicted {CAMERA_TO_USE} {MODE}")
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()
