import os
import re
import random
import argparse
import tarfile
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
    Dense, Dropout, BatchNormalization
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
BASE_DATASET = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_ROOT = os.path.join(BASE_DATASET, "by_person")
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

TARGET_EXERCISE = "E3"
CAMERAS_TO_RUN = ["C0", "C1", "C2", "ALL_COMPLETE"]

pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    score_map[(row["exercise"], row["person"], row["trial"])] = float(row["mean"])


def reset_seed():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_split(split_name, camera_mode):
    split_dir = os.path.join(SPLIT_ROOT, split_name)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    segment_samples = []
    bad_files = []
    skipped_not_target = 0

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

        if camera_mode != "ALL_COMPLETE" and camera != camera_mode:
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

    incomplete_trials = []

    if camera_mode == "ALL_COMPLETE":
        trial_to_cameras = defaultdict(set)

        for sample in segment_samples:
            trial_key = (
                sample["exercise"],
                sample["person"],
                sample["trial"]
            )
            trial_to_cameras[trial_key].add(sample["camera"])

        required_cameras = {"C0", "C1", "C2"}
        valid_trials = set()

        for trial_key, cams in trial_to_cameras.items():
            if required_cameras.issubset(cams):
                valid_trials.add(trial_key)
            else:
                incomplete_trials.append({
                    "split": split_name,
                    "exercise": trial_key[0],
                    "person": trial_key[1],
                    "trial": trial_key[2],
                    "available_cameras": ",".join(sorted(cams)),
                    "missing_cameras": ",".join(sorted(required_cameras - cams))
                })

        filtered_samples = []

        for sample in segment_samples:
            trial_key = (
                sample["exercise"],
                sample["person"],
                sample["trial"]
            )

            if trial_key in valid_trials:
                filtered_samples.append(sample)

        segment_samples = filtered_samples

    grouped = defaultdict(list)

    for sample in segment_samples:
        group_key = (
            sample["exercise"],
            sample["person"],
            sample["trial"],
            sample["camera"]
        )
        grouped[group_key].append(
            (sample["segment"], sample["x"], sample["y"], sample["file_name"])
        )

    trial_samples = []

    for group_key, segs in grouped.items():
        segs = sorted(segs, key=lambda item: item[0])
        arrays = [arr for _, arr, _, _ in segs]

        if len(arrays) == 0:
            continue

        full_sequence = np.concatenate(arrays, axis=0)
        label = float(segs[0][2])
        base_file_name = segs[0][3]

        exercise, person, trial, camera = group_key

        trial_samples.append({
            "file_name": base_file_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "x": full_sequence.astype(np.float32),
            "y": label
        })

    return trial_samples, bad_files, skipped_not_target, incomplete_trials


def samples_to_df(samples, split_name):
    rows = []

    for s in samples:
        rows.append({
            "split": split_name,
            "exercise": s["exercise"],
            "person": s["person"],
            "trial": s["trial"],
            "camera": s["camera"],
            "score": s["y"],
            "performance": "good" if s["y"] >= 4.0 else "bad"
        })

    return pd.DataFrame(rows)


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


def prepare_xy(sample_list):
    X = []
    y = []
    meta = []

    for sample in sample_list:
        seq_3d = sample["x"].copy()

        seq_3d = center_skeleton(seq_3d)
        seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)

        seq_2d = smooth_sequence(seq_2d, window_size=5)
        seq_2d = resample_sequence(seq_2d, TARGET_LEN)
        seq_2d = normalize_per_sample(seq_2d)
        seq_2d = add_velocity_features(seq_2d)

        X.append(seq_2d)
        y.append(float(sample["y"]))

        meta.append({
            "exercise": sample["exercise"],
            "person": sample["person"],
            "trial": sample["trial"],
            "camera": sample["camera"]
        })

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        meta
    )


def build_model(feature_dim):
    seq_input = Input(shape=(TARGET_LEN, feature_dim), name="sequence_input")

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
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    return model


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

        train_r2_value = r2_score(self.y_train_scaled, train_pred)
        val_r2_value = r2_score(self.y_val_scaled, val_pred)

        train_pcc_value = pearsonr(self.y_train_scaled, train_pred)[0] if len(self.y_train_scaled) > 1 else np.nan
        val_pcc_value = pearsonr(self.y_val_scaled, val_pred)[0] if len(self.y_val_scaled) > 1 else np.nan

        self.train_r2.append(train_r2_value)
        self.val_r2.append(val_r2_value)
        self.train_pcc.append(train_pcc_value)
        self.val_pcc.append(val_pcc_value)


def get_ylim(values, pad_ratio=0.15):
    vmin = min(values)
    vmax = max(values)
    pad = (vmax - vmin) * pad_ratio if vmax != vmin else 0.1
    return vmin - pad, vmax + pad


def save_plots(history, epoch_metrics, test_loss, test_mae_scaled, test_rmse_scaled, r2, pcc, y_test, y_pred, output_dir, title_prefix):
    loss_values = history.history["loss"]
    val_loss_values = history.history["val_loss"]
    all_loss = loss_values + val_loss_values + [test_loss]
    loss_ymin, loss_ymax = get_ylim(all_loss)

    plt.figure(figsize=(9, 5))
    plt.plot(loss_values, label="Train", linewidth=2)
    plt.plot(val_loss_values, label="Validation", linewidth=2)
    plt.axhline(y=test_loss, linestyle="-.", linewidth=2, label=f"Test Loss={test_loss:.4f}")
    plt.title(f"{title_prefix} Regression Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(loss_ymin, loss_ymax)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
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

    plt.suptitle(f"{title_prefix} RMSE & MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_mae.png"), dpi=300, bbox_inches="tight")
    plt.close()

    all_r2 = epoch_metrics.train_r2 + epoch_metrics.val_r2 + [r2, 0, 1]
    r2_ymin, r2_ymax = get_ylim(all_r2, pad_ratio=0.10)

    plt.figure(figsize=(9, 5))
    plt.plot(epoch_metrics.train_r2, label="Train", linewidth=2)
    plt.plot(epoch_metrics.val_r2, label="Validation", linewidth=2)
    plt.axhline(y=r2, linestyle="-.", linewidth=2, label=f"Test R2={r2:.4f}")
    plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
    plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="Baseline")
    plt.title(f"{title_prefix} R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.ylim(r2_ymin, r2_ymax)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r2_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    all_pcc = epoch_metrics.train_pcc + epoch_metrics.val_pcc + [pcc, 0, 1]
    pcc_ymin, pcc_ymax = get_ylim(all_pcc, pad_ratio=0.10)

    plt.figure(figsize=(9, 5))
    plt.plot(epoch_metrics.train_pcc, label="Train", linewidth=2)
    plt.plot(epoch_metrics.val_pcc, label="Validation", linewidth=2)
    plt.axhline(y=pcc, linestyle="-.", linewidth=2, label=f"Test PCC={pcc:.4f}")
    plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
    plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="No correlation")
    plt.title(f"{title_prefix} PCC")
    plt.xlabel("Epoch")
    plt.ylabel("PCC")
    plt.ylim(pcc_ymin, pcc_ymax)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([1, 5], [1, 5], "r--")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{title_prefix} True vs Predicted")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    image_files = [
        "loss_curve.png",
        "rmse_mae.png",
        "r2_curve.png",
        "pcc_curve.png",
        "true_vs_predicted.png"
    ]

    tar_path = os.path.join(output_dir, "images.tar.gz")

    with tarfile.open(tar_path, "w:gz") as tar:
        for img in image_files:
            img_path = os.path.join(output_dir, img)
            if os.path.exists(img_path):
                tar.add(img_path, arcname=img)


def run_experiment(camera_mode):
    reset_seed()

    if camera_mode == "ALL_COMPLETE":
        folder_name = f"results_single_{TARGET_EXERCISE}_all_cameras_complete_velocity_only"
        title_prefix = f"{TARGET_EXERCISE} All Cameras Complete"
        target_camera_text = "C0+C1+C2 complete only"
    else:
        folder_name = f"results_single_{TARGET_EXERCISE}_{camera_mode}_velocity_only"
        title_prefix = f"{TARGET_EXERCISE} {camera_mode}"
        target_camera_text = camera_mode

    output_dir = f"/mvdlph/shahd/MVCNNPH/{folder_name}/len{TARGET_LEN}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=================================================")
    print(f"Running: {title_prefix} Velocity Only")
    print("Output:", output_dir)
    print("=================================================\n")

    fit_samples, bad_train, skipped_train, incomplete_train = load_split("train", camera_mode)
    val_samples, bad_valid, skipped_valid, incomplete_valid = load_split("valid", camera_mode)
    test_samples, bad_test, skipped_test, incomplete_test = load_split("test", camera_mode)

    bad_files = bad_train + bad_valid + bad_test
    incomplete_trials = incomplete_train + incomplete_valid + incomplete_test

    with open(os.path.join(output_dir, "bad_files.txt"), "w", encoding="utf-8") as f:
        for item in bad_files:
            f.write(str(item) + "\n")

    if camera_mode == "ALL_COMPLETE":
        pd.DataFrame(incomplete_trials).to_csv(
            os.path.join(output_dir, "incomplete_camera_trials.csv"),
            index=False
        )

    print("Fit samples:", len(fit_samples))
    print("Validation samples:", len(val_samples))
    print("Test samples:", len(test_samples))
    print("Bad files:", len(bad_files))
    print("Skipped non-target files:", skipped_train + skipped_valid + skipped_test)
    print("Incomplete trials:", len(incomplete_trials))

    if len(fit_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        with open(os.path.join(output_dir, "ERROR_empty_split.txt"), "w", encoding="utf-8") as f:
            f.write("One split has zero samples.\n")
            f.write(f"Fit samples = {len(fit_samples)}\n")
            f.write(f"Validation samples = {len(val_samples)}\n")
            f.write(f"Test samples = {len(test_samples)}\n")
        print("ERROR: One split has zero samples. Skipping this experiment.")
        return

    fit_dist_df = samples_to_df(fit_samples, "train")
    val_dist_df = samples_to_df(val_samples, "valid")
    test_dist_df = samples_to_df(test_samples, "test")

    split_distribution_df = pd.concat(
        [fit_dist_df, val_dist_df, test_dist_df],
        ignore_index=True
    )
    split_distribution_df.to_csv(os.path.join(output_dir, "split_distribution.csv"), index=False)

    person_summary_df = split_distribution_df.groupby(["split", "person"]).agg(
        num_samples=("score", "count"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        good_count=("performance", lambda x: (x == "good").sum()),
        bad_count=("performance", lambda x: (x == "bad").sum())
    ).reset_index()

    person_summary_df.to_csv(os.path.join(output_dir, "person_split_summary.csv"), index=False)

    camera_summary_df = split_distribution_df.groupby(["split", "camera"]).agg(
        num_samples=("score", "count"),
        mean_score=("score", "mean"),
        good_count=("performance", lambda x: (x == "good").sum()),
        bad_count=("performance", lambda x: (x == "bad").sum())
    ).reset_index()

    camera_summary_df.to_csv(os.path.join(output_dir, "camera_split_summary.csv"), index=False)

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

    model = build_model(feature_dim)
    model.summary()

    epoch_metrics = RegressionMetricsCallback(
        X_fit,
        y_fit_scaled,
        X_val,
        y_val_scaled
    )

    checkpoint_path = os.path.join(output_dir, "best_model.keras")

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
        X_fit,
        y_fit_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=EPOCHS,
        batch_size=256,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    with open(os.path.join(output_dir, "training_log.txt"), "w", encoding="utf-8") as f:
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

    y_pred_scaled = model.predict(X_test).flatten()

    test_loss, test_mae_scaled, test_rmse_scaled = model.evaluate(
        X_test,
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
            "camera": meta["camera"],
            "true_score": true_value,
            "predicted_score": pred_value,
            "abs_error": abs(true_value - pred_value)
        })

    predictions_df = pd.DataFrame(pred_rows)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    worst_predictions_df = predictions_df.sort_values("abs_error", ascending=False).head(30)
    worst_predictions_df.to_csv(os.path.join(output_dir, "worst_30_predictions.csv"), index=False)

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
    per_person_df.to_csv(os.path.join(output_dir, "per_person_scores.csv"), index=False)

    per_camera_rows = []

    for camera, group in predictions_df.groupby("camera"):
        true = group["true_score"].values
        pred = group["predicted_score"].values
        camera_pcc = pearsonr(true, pred)[0] if len(group) > 1 else np.nan

        per_camera_rows.append({
            "camera": camera,
            "num_test_trials": len(group),
            "mean_true_score": np.mean(true),
            "mean_predicted_score": np.mean(pred),
            "MAE": mean_absolute_error(true, pred),
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "R2": r2_score(true, pred) if len(group) > 1 else np.nan,
            "PCC": camera_pcc
        })

    per_camera_df = pd.DataFrame(per_camera_rows).sort_values("camera")
    per_camera_df.to_csv(os.path.join(output_dir, "per_camera_scores.csv"), index=False)

    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"===== Single-view {TARGET_EXERCISE} {target_camera_text} Velocity Only Results =====\n")
        f.write(f"Target exercise = {TARGET_EXERCISE}\n")
        f.write(f"Target camera mode = {target_camera_text}\n")
        f.write(f"Target_len = {TARGET_LEN}\n")
        f.write(f"Epochs requested = {EPOCHS}\n")
        f.write(f"Epochs trained = {len(history.history['loss'])}\n")
        f.write("Loss = MSE\n")
        f.write("Learning rate = 2e-5\n")
        f.write("Batch size = 256\n")
        f.write("Filters = 16, 24, 24, 32\n")
        f.write("Dropout = 0.10\n")
        f.write("Added features = velocity only\n")
        f.write("Exercise embedding = No\n")
        f.write("Oversampling = No\n")
        f.write("Replacement = No\n")
        f.write("Augmentation = No\n")

        if camera_mode == "ALL_COMPLETE":
            f.write("Complete-camera rule = if any of C0/C1/C2 is missing, the full trial is skipped from all cameras\n")

        f.write("Images archive = images.tar.gz\n\n")

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
        f.write(f"Skipped non-target files = {skipped_train + skipped_valid + skipped_test}\n")
        f.write(f"Incomplete camera trials = {len(incomplete_trials)}\n\n")

        f.write("===== Split Person Summary =====\n")
        f.write(person_summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("===== Camera Split Summary =====\n")
        f.write(camera_summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("===== Per-person scores =====\n")
        f.write(per_person_df.to_string(index=False))
        f.write("\n\n")

        f.write("===== Per-camera scores =====\n")
        f.write(per_camera_df.to_string(index=False))
        f.write("\n\n")

        f.write("===== Worst 30 predictions =====\n")
        f.write(worst_predictions_df.to_string(index=False))

    np.savez(
        os.path.join(output_dir, "plot_data.npz"),
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

    save_plots(
        history,
        epoch_metrics,
        test_loss,
        test_mae_scaled,
        test_rmse_scaled,
        r2,
        pcc,
        y_test,
        y_pred,
        output_dir,
        title_prefix
    )

    print("\nSaved files in:", output_dir)
    print("metrics.txt")
    print("training_log.txt")
    print("predictions.csv")
    print("worst_30_predictions.csv")
    print("per_person_scores.csv")
    print("per_camera_scores.csv")
    print("split_distribution.csv")
    print("person_split_summary.csv")
    print("camera_split_summary.csv")
    print("plot_data.npz")
    print("loss_curve.png")
    print("rmse_mae.png")
    print("r2_curve.png")
    print("pcc_curve.png")
    print("true_vs_predicted.png")
    print("images.tar.gz")
    print("bad_files.txt")

    if camera_mode == "ALL_COMPLETE":
        print("incomplete_camera_trials.csv")

    print("best_model.keras")


for camera_mode in CAMERAS_TO_RUN:
    run_experiment(camera_mode)
