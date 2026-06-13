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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization, Embedding, Flatten, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


parser = argparse.ArgumentParser()
parser.add_argument("--target_len", type=int, default=150)
parser.add_argument("--epochs", type=int, default=500)
args = parser.parse_args()

TARGET_LEN = args.target_len
EPOCHS = args.epochs

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DATASET = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_ROOT = os.path.join(BASE_DATASET, "by_person")
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results_multiview_early_fusion_mse_aug_oversample/len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Experiment: Multi-view Early Fusion MSE + Full Train Augmentation + Oversampling Replacement")
print("Target length:", TARGET_LEN)
print("Epochs:", EPOCHS)
print("Output:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    score_map[(row["exercise"], row["person"], row["trial"])] = float(row["mean"])

pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")


def exercise_to_id(exercise):
    return int(exercise.replace("E", ""))


LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6),
    (11, 14), (12, 15), (13, 16)
]


def augment_sequence(seq_3d):
    seq = seq_3d.copy().astype(np.float32)

    if random.random() < 0.60:
        seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)

    if random.random() < 0.50:
        scale = random.uniform(0.95, 1.05)
        seq *= scale

    if random.random() < 0.50:
        shift = random.randint(-5, 5)
        seq = np.roll(seq, shift, axis=0)

    if random.random() < 0.50:
        angle = np.deg2rad(random.uniform(-8, 8))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        x = seq[:, :, 0].copy()
        z = seq[:, :, 2].copy()

        seq[:, :, 0] = cos_a * x - sin_a * z
        seq[:, :, 2] = sin_a * x + cos_a * z

    if random.random() < 0.50:
        seq[:, :, 0] *= -1
        for left, right in LEFT_RIGHT_PAIRS:
            seq[:, [left, right], :] = seq[:, [right, left], :]

    return seq.astype(np.float32)


def augment_multiview_sample(sample, suffix):
    return {
        "exercise": sample["exercise"],
        "person": sample["person"],
        "trial": sample["trial"],
        "x_c0": augment_sequence(sample["x_c0"]),
        "x_c1": augment_sequence(sample["x_c1"]),
        "x_c2": augment_sequence(sample["x_c2"]),
        "y": float(sample["y"]),
        "is_augmented": 1,
        "aug_type": suffix
    }


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


def load_multiview_split(split_name):
    split_dir = os.path.join(SPLIT_ROOT, split_name)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    segment_samples = []
    bad_files = []

    for file_name in sorted(os.listdir(split_dir)):
        if not file_name.endswith(".npz"):
            continue

        match = pattern.search(file_name)
        if not match:
            bad_files.append((file_name, "filename pattern not matched"))
            continue

        exercise, person, trial, camera, seg_id = match.groups()
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
            bad_files.append((file_name, str(keypoints_3d.shape)))
            continue

        segment_samples.append({
            "file_name": file_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "segment": int(seg_id),
            "x": keypoints_3d.astype(np.float32),
            "y": score_map[label_key]
        })

    camera_grouped = defaultdict(list)

    for sample in segment_samples:
        key = (
            sample["exercise"],
            sample["person"],
            sample["trial"],
            sample["camera"]
        )
        camera_grouped[key].append(
            (sample["segment"], sample["x"], sample["y"])
        )

    trial_camera_sequences = defaultdict(dict)

    for key, segs in camera_grouped.items():
        exercise, person, trial, camera = key
        segs = sorted(segs, key=lambda item: item[0])
        arrays = [arr for _, arr, _ in segs]

        if len(arrays) == 0:
            continue

        full_sequence = np.concatenate(arrays, axis=0)
        label = float(segs[0][2])

        trial_key = (exercise, person, trial)
        trial_camera_sequences[trial_key][camera] = {
            "x": full_sequence.astype(np.float32),
            "y": label
        }

    required_cameras = {"C0", "C1", "C2"}
    multiview_samples = []
    incomplete_trials = []

    for trial_key, cam_dict in trial_camera_sequences.items():
        exercise, person, trial = trial_key
        available = set(cam_dict.keys())

        if not required_cameras.issubset(available):
            incomplete_trials.append({
                "split": split_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "available_cameras": ",".join(sorted(available)),
                "missing_cameras": ",".join(sorted(required_cameras - available))
            })
            continue

        label = cam_dict["C0"]["y"]

        multiview_samples.append({
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "x_c0": cam_dict["C0"]["x"],
            "x_c1": cam_dict["C1"]["x"],
            "x_c2": cam_dict["C2"]["x"],
            "y": label,
            "is_augmented": 0,
            "aug_type": "original"
        })

    return multiview_samples, bad_files, incomplete_trials


def augment_full_training_data(samples):
    augmented = []

    for i, sample in enumerate(samples):
        aug_sample = augment_multiview_sample(sample, f"full_train_aug_{i}")
        augmented.append(aug_sample)

    print("\n===== Full Train Augmentation Summary =====")
    print("Original train samples:", len(samples))
    print("Full train augmented samples added:", len(augmented))
    print("Train after full augmentation:", len(samples) + len(augmented))

    return list(samples) + augmented


def oversample_low_scores_with_replacement(samples):
    low = [s for s in samples if s["y"] < 3.0]
    mid_low = [s for s in samples if 3.0 <= s["y"] < 4.0]
    high = [s for s in samples if s["y"] >= 4.0]

    target_count = len(high)
    augmented = []

    for group_name, group in [("low", low), ("mid_low", mid_low)]:
        if len(group) == 0:
            continue

        needed = max(0, target_count - len(group))

        for i in range(needed):
            base_sample = random.choice(group)
            aug_sample = augment_multiview_sample(
                base_sample,
                f"oversample_{group_name}_{i}"
            )
            augmented.append(aug_sample)

    summary_df = pd.DataFrame([
        {"group": "low_<3", "count_before": len(low), "target_count": target_count},
        {"group": "mid_low_3_to_4", "count_before": len(mid_low), "target_count": target_count},
        {"group": "high_>=4", "count_before": len(high), "target_count": target_count},
        {"group": "oversampled_added", "count_before": len(augmented), "target_count": target_count},
        {"group": "final_train", "count_before": len(samples) + len(augmented), "target_count": target_count},
    ])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "oversampling_summary.csv"), index=False)

    print("\n===== Oversampling Replacement Summary =====")
    print("Low score samples (<3):", len(low))
    print("Mid-low score samples [3,4):", len(mid_low))
    print("High score samples (>=4):", len(high))
    print("Target count:", target_count)
    print("Oversampled augmented samples added:", len(augmented))
    print("Final train samples:", len(samples) + len(augmented))

    return list(samples) + augmented


fit_samples, bad_train, incomplete_train = load_multiview_split("train")
val_samples, bad_valid, incomplete_valid = load_multiview_split("valid")
test_samples, bad_test, incomplete_test = load_multiview_split("test")

original_train_count = len(fit_samples)

fit_samples = augment_full_training_data(fit_samples)
after_full_aug_count = len(fit_samples)

fit_samples = oversample_low_scores_with_replacement(fit_samples)
after_oversample_count = len(fit_samples)

bad_files = bad_train + bad_valid + bad_test
incomplete_trials = incomplete_train + incomplete_valid + incomplete_test

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w", encoding="utf-8") as f:
    for item in bad_files:
        f.write(str(item) + "\n")

pd.DataFrame(incomplete_trials).to_csv(
    os.path.join(OUTPUT_DIR, "incomplete_multiview_trials.csv"),
    index=False
)

print("\nFit multiview samples:", len(fit_samples))
print("Validation multiview samples:", len(val_samples))
print("Test multiview samples:", len(test_samples))
print("Bad files:", len(bad_files))
print("Incomplete multiview trials:", len(incomplete_trials))


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
            "is_augmented": s.get("is_augmented", 0),
            "aug_type": s.get("aug_type", "original")
        })
    return pd.DataFrame(rows)


fit_dist_df = samples_to_df(fit_samples, "train")
val_dist_df = samples_to_df(val_samples, "valid")
test_dist_df = samples_to_df(test_samples, "test")

split_distribution_df = pd.concat(
    [fit_dist_df, val_dist_df, test_dist_df],
    ignore_index=True
)
split_distribution_df.to_csv(os.path.join(OUTPUT_DIR, "split_distribution.csv"), index=False)

person_summary_df = split_distribution_df.groupby(["split", "person"]).agg(
    num_samples=("score", "count"),
    mean_score=("score", "mean"),
    min_score=("score", "min"),
    max_score=("score", "max"),
    good_count=("performance", lambda x: (x == "good").sum()),
    bad_count=("performance", lambda x: (x == "bad").sum()),
    augmented_count=("is_augmented", "sum")
).reset_index()

person_summary_df.to_csv(os.path.join(OUTPUT_DIR, "person_split_summary.csv"), index=False)

exercise_summary_df = split_distribution_df.groupby(["split", "exercise"]).agg(
    num_samples=("score", "count"),
    mean_score=("score", "mean"),
    good_count=("performance", lambda x: (x == "good").sum()),
    bad_count=("performance", lambda x: (x == "bad").sum()),
    augmented_count=("is_augmented", "sum")
).reset_index()

exercise_summary_df.to_csv(os.path.join(OUTPUT_DIR, "exercise_split_summary.csv"), index=False)


def prepare_xy(sample_list):
    X = []
    X_exercise = []
    y = []
    meta = []

    for sample in sample_list:
        c0 = preprocess_one_view(sample["x_c0"])
        c1 = preprocess_one_view(sample["x_c1"])
        c2 = preprocess_one_view(sample["x_c2"])

        fused = np.concatenate([c0, c1, c2], axis=1)

        X.append(fused)
        X_exercise.append(exercise_to_id(sample["exercise"]))
        y.append(float(sample["y"]))

        meta.append({
            "exercise": sample["exercise"],
            "person": sample["person"],
            "trial": sample["trial"],
            "is_augmented": sample.get("is_augmented", 0),
            "aug_type": sample.get("aug_type", "original")
        })

    return (
        np.array(X, dtype=np.float32),
        np.array(X_exercise, dtype=np.int32),
        np.array(y, dtype=np.float32),
        meta
    )


X_fit, X_fit_ex, y_fit, fit_meta = prepare_xy(fit_samples)
X_val, X_val_ex, y_val, val_meta = prepare_xy(val_samples)
X_test, X_test_ex, y_test, test_meta = prepare_xy(test_samples)

print("X_fit shape:", X_fit.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("Exercise input shape:", X_fit_ex.shape)

y_scaler = StandardScaler()
y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

feature_dim = X_fit.shape[2]
num_exercises = 10

seq_input = Input(shape=(TARGET_LEN, feature_dim), name="multiview_sequence_input")

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

exercise_input = Input(shape=(1,), name="exercise_input")
e = Embedding(input_dim=num_exercises, output_dim=4, name="exercise_embedding")(exercise_input)
e = Flatten()(e)

merged = Concatenate()([x, e])
merged = Dense(32, activation="relu")(merged)
merged = Dropout(0.10)(merged)
output = Dense(1)(merged)

model = Model(inputs=[seq_input, exercise_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

model.summary()


class RegressionMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, X_train_ex, y_train_scaled, X_val, X_val_ex, y_val_scaled):
        super().__init__()
        self.X_train = X_train
        self.X_train_ex = X_train_ex
        self.y_train_scaled = y_train_scaled
        self.X_val = X_val
        self.X_val_ex = X_val_ex
        self.y_val_scaled = y_val_scaled
        self.train_r2 = []
        self.val_r2 = []
        self.train_pcc = []
        self.val_pcc = []

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict([self.X_train, self.X_train_ex], verbose=0).flatten()
        val_pred = self.model.predict([self.X_val, self.X_val_ex], verbose=0).flatten()

        train_r2_value = r2_score(self.y_train_scaled, train_pred)
        val_r2_value = r2_score(self.y_val_scaled, val_pred)

        train_pcc_value = pearsonr(self.y_train_scaled, train_pred)[0] if len(self.y_train_scaled) > 1 else np.nan
        val_pcc_value = pearsonr(self.y_val_scaled, val_pred)[0] if len(self.y_val_scaled) > 1 else np.nan

        self.train_r2.append(train_r2_value)
        self.val_r2.append(val_r2_value)
        self.train_pcc.append(train_pcc_value)
        self.val_pcc.append(val_pcc_value)


epoch_metrics = RegressionMetricsCallback(
    X_fit, X_fit_ex, y_fit_scaled,
    X_val, X_val_ex, y_val_scaled
)

checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")

callbacks = [
    epoch_metrics,
    EarlyStopping(
        monitor="val_loss",
        patience=80,
        min_delta=0.00005,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    [X_fit, X_fit_ex],
    y_fit_scaled,
    validation_data=([X_val, X_val_ex], y_val_scaled),
    epochs=EPOCHS,
    batch_size=256,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i + 1}\n")
        f.write(f"Train Loss = {history.history['loss'][i]}\n")
        f.write(f"Val Loss = {history.history['val_loss'][i]}\n")
        f.write(f"Train MAE = {history.history['mae'][i]}\n")
        f.write(f"Val MAE = {history.history['val_mae'][i]}\n")
        f.write(f"Train RMSE = {history.history['rmse'][i]}\n")
        f.write(f"Val RMSE = {history.history['val_rmse'][i]}\n")
        f.write(f"Train R2 = {epoch_metrics.train_r2[i]}\n")
        f.write(f"Val R2 = {epoch_metrics.val_r2[i]}\n")
        f.write(f"Train PCC = {epoch_metrics.train_pcc[i]}\n")
        f.write(f"Val PCC = {epoch_metrics.val_pcc[i]}\n\n")

y_pred_scaled = model.predict([X_test, X_test_ex]).flatten()

test_loss, test_mae_scaled, test_rmse_scaled = model.evaluate(
    [X_test, X_test_ex],
    y_test_scaled,
    verbose=0
)

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.clip(y_pred, 1.0, 5.0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

if len(y_test) > 1:
    pcc, pcc_pvalue = pearsonr(y_test, y_pred)
else:
    pcc, pcc_pvalue = np.nan, np.nan

print("\n===== Overall Test Results =====")
print("MAE =", mae)
print("RMSE =", rmse)
print("R2 =", r2)
print("PCC =", pcc)
print("PCC p-value =", pcc_pvalue)
print("Test Loss =", test_loss)
print("Test MAE scaled =", test_mae_scaled)
print("Test RMSE scaled =", test_rmse_scaled)

pred_rows = []

for meta, true_value, pred_value in zip(test_meta, y_test, y_pred):
    pred_rows.append({
        "exercise": meta["exercise"],
        "person": meta["person"],
        "trial": meta["trial"],
        "true_score": true_value,
        "predicted_score": pred_value,
        "abs_error": abs(true_value - pred_value)
    })

predictions_df = pd.DataFrame(pred_rows)
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

worst_predictions_df = predictions_df.sort_values("abs_error", ascending=False).head(30)
worst_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "worst_30_predictions.csv"), index=False)

per_exercise_rows = []
for exercise, group in predictions_df.groupby("exercise"):
    true = group["true_score"].values
    pred = group["predicted_score"].values
    ex_pcc = pearsonr(true, pred)[0] if len(group) > 1 else np.nan

    per_exercise_rows.append({
        "exercise": exercise,
        "num_test_trials": len(group),
        "mean_true_score": np.mean(true),
        "mean_predicted_score": np.mean(pred),
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred) if len(group) > 1 else np.nan,
        "PCC": ex_pcc
    })

per_exercise_df = pd.DataFrame(per_exercise_rows).sort_values("exercise")
per_exercise_df.to_csv(os.path.join(OUTPUT_DIR, "per_exercise_scores.csv"), index=False)

per_person_rows = []
for person, group in predictions_df.groupby("person"):
    true = group["true_score"].values
    pred = group["predicted_score"].values
    person_pcc = pearsonr(true, pred)[0] if len(group) > 1 else np.nan

    per_person_rows.append({
        "person": person,
        "num_test_trials": len(group),
        "mean_true_score": np.mean(true),
        "mean_predicted_score": np.mean(pred),
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred) if len(group) > 1 else np.nan,
        "PCC": person_pcc
    })

per_person_df = pd.DataFrame(per_person_rows).sort_values("person")
per_person_df.to_csv(os.path.join(OUTPUT_DIR, "per_person_scores.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write("===== Multi-view Early Fusion MSE + Augmentation + Oversampling Results =====\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"Epochs requested = {EPOCHS}\n")
    f.write(f"Epochs trained = {len(history.history['loss'])}\n")
    f.write("Fusion type = Early fusion\n")
    f.write("Views = C0 + C1 + C2\n")
    f.write("Fusion method = concatenate along feature dimension\n")
    f.write("Input per view = 102 features\n")
    f.write("Fused input features = 306\n")
    f.write("Loss = MSE\n")
    f.write("Learning rate = 2e-5\n")
    f.write("Batch size = 256\n")
    f.write("Filters = 16, 24, 24, 32\n")
    f.write("Dropout = 0.10\n")
    f.write("Added features = velocity only\n")
    f.write("Exercise embedding = Yes, embedding_dim=4\n")
    f.write("Sample weighting = No\n")
    f.write("Full train augmentation = Yes\n")
    f.write("Oversampling with replacement = Yes\n")
    f.write("Augmentation = jitter, scaling, temporal shift, small rotation, mirroring\n")
    f.write("Incomplete camera rule = if any camera is missing, the full trial is skipped\n")
    f.write(f"Original train samples = {original_train_count}\n")
    f.write(f"After full train augmentation = {after_full_aug_count}\n")
    f.write(f"After oversampling = {after_oversample_count}\n\n")

    f.write("===== Overall Test Metrics =====\n")
    f.write(f"Overall MAE = {mae}\n")
    f.write(f"Overall RMSE = {rmse}\n")
    f.write(f"Overall R2 = {r2}\n")
    f.write(f"Overall PCC = {pcc}\n")
    f.write(f"Overall PCC p-value = {pcc_pvalue}\n")
    f.write(f"Test Loss = {test_loss}\n")
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
    f.write(f"Incomplete multiview trials = {len(incomplete_trials)}\n\n")

    f.write("===== Split Person Summary =====\n")
    f.write(person_summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Split Exercise Summary =====\n")
    f.write(exercise_summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Per-exercise scores =====\n")
    f.write(per_exercise_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Per-person scores =====\n")
    f.write(per_person_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== Worst 30 predictions =====\n")
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
    test_loss=np.array([test_loss]),
    test_mae_scaled=np.array([test_mae_scaled]),
    test_rmse_scaled=np.array([test_rmse_scaled]),
    test_r2=np.array([r2]),
    test_pcc=np.array([pcc]),
    y_test=y_test,
    y_pred=y_pred
)


def get_ylim(values, pad_ratio=0.15):
    vmin = min(values)
    vmax = max(values)
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


loss_values = history.history["loss"]
val_loss_values = history.history["val_loss"]
all_loss = loss_values + val_loss_values + [test_loss]
loss_ymin, loss_ymax = get_ylim(all_loss)

plt.figure(figsize=(9, 5))
plt.plot(loss_values, label="Train", linewidth=2)
plt.plot(val_loss_values, label="Validation", linewidth=2)
plt.axhline(y=test_loss, linestyle="-.", linewidth=2, label=f"Test Loss={test_loss:.4f}")
plt.title("Multi-view Early Fusion Regression Loss (MSE)")
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

plt.suptitle("Multi-view Early Fusion RMSE & MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_mae.png"), dpi=300, bbox_inches="tight")
plt.close()

all_r2 = epoch_metrics.train_r2 + epoch_metrics.val_r2 + [r2, 0, 1]
r2_ymin, r2_ymax = get_ylim(all_r2, pad_ratio=0.10)

plt.figure(figsize=(9, 5))
plt.plot(epoch_metrics.train_r2, label="Train", linewidth=2)
plt.plot(epoch_metrics.val_r2, label="Validation", linewidth=2)
plt.axhline(y=r2, linestyle="-.", linewidth=2, label=f"Test R2={r2:.4f}")
plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="Baseline")
plt.title("Multi-view Early Fusion R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.ylim(r2_ymin, r2_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "r2_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

all_pcc = epoch_metrics.train_pcc + epoch_metrics.val_pcc + [pcc, 0, 1]
pcc_ymin, pcc_ymax = get_ylim(all_pcc, pad_ratio=0.10)

plt.figure(figsize=(9, 5))
plt.plot(epoch_metrics.train_pcc, label="Train", linewidth=2)
plt.plot(epoch_metrics.val_pcc, label="Validation", linewidth=2)
plt.axhline(y=pcc, linestyle="-.", linewidth=2, label=f"Test PCC={pcc:.4f}")
plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="No correlation")
plt.title("Multi-view Early Fusion PCC")
plt.xlabel("Epoch")
plt.ylabel("PCC")
plt.ylim(pcc_ymin, pcc_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pcc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([1, 5], [1, 5], "r--")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Multi-view Early Fusion True vs Predicted")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved files in:", OUTPUT_DIR)
print("metrics.txt")
print("training_log.txt")
print("predictions.csv")
print("worst_30_predictions.csv")
print("per_exercise_scores.csv")
print("per_person_scores.csv")
print("split_distribution.csv")
print("person_split_summary.csv")
print("exercise_split_summary.csv")
print("oversampling_summary.csv")
print("plot_data.npz")
print("loss_curve.png")
print("rmse_mae.png")
print("r2_curve.png")
print("pcc_curve.png")
print("true_vs_predicted.png")
print("bad_files.txt")
print("incomplete_multiview_trials.csv")
print("best_model.keras")
