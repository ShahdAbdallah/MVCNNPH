
import os, re, random, argparse, tarfile
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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument("--target_len", type=int, default=120)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--exercise", type=str, default="E1")
parser.add_argument("--batch_name", type=str, default="batch_train_val_test_checked_v2")
args = parser.parse_args()

TARGET_LEN = args.target_len
EPOCHS = args.epochs
TARGET_EXERCISE = args.exercise.strip()
FUSION_TYPE = "late"
FUSION_FOLDER = "late_fusion"
FUSION_TITLE = "Late Fusion"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DATASET = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_ROOT = os.path.join(BASE_DATASET, "by_person")
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"
REQUIRED_CAMERAS = {"C0", "C1", "C2"}

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results_multiview_{TARGET_EXERCISE}_{FUSION_FOLDER}_velocity_only_{args.batch_name}/len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Experiment: Multi-view {TARGET_EXERCISE} {FUSION_TITLE} Velocity Only")
print("Target length:", TARGET_LEN)
print("Epochs:", EPOCHS)
print("Output:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()
score_map = {(r["exercise"], r["person"], r["trial"]): float(r["mean"]) for _, r in labels_df.iterrows()}
pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")


def center_skeleton(seq_3d, root_idx=0):
    return seq_3d - seq_3d[:, root_idx:root_idx + 1, :]


def smooth_sequence(seq_2d, window_size=5):
    pad = window_size // 2
    padded = np.pad(seq_2d, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(seq_2d, dtype=np.float32)
    for t in range(seq_2d.shape[0]):
        out[t] = padded[t:t + window_size].mean(axis=0)
    return out


def resample_sequence(seq_2d, target_len):
    old_len = seq_2d.shape[0]
    if old_len == target_len:
        return seq_2d.astype(np.float32)
    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, seq_2d.shape[1]), dtype=np.float32)
    for j in range(seq_2d.shape[1]):
        out[:, j] = np.interp(new_idx, old_idx, seq_2d[:, j])
    return out


def normalize_per_sample(seq_2d):
    return (seq_2d - seq_2d.mean(axis=0, keepdims=True)) / (seq_2d.std(axis=0, keepdims=True) + 1e-8)


def add_velocity_features(seq_2d):
    vel = np.zeros_like(seq_2d, dtype=np.float32)
    vel[1:] = seq_2d[1:] - seq_2d[:-1]
    return np.concatenate([seq_2d, vel], axis=1)


def preprocess_one_view(seq_3d):
    seq_3d = center_skeleton(seq_3d)
    seq_2d = seq_3d.reshape(seq_3d.shape[0], -1)
    seq_2d = smooth_sequence(seq_2d, 5)
    seq_2d = resample_sequence(seq_2d, TARGET_LEN)
    seq_2d = normalize_per_sample(seq_2d)
    seq_2d = add_velocity_features(seq_2d)
    return seq_2d.astype(np.float32)


def load_multiview_split(split_name):
    split_dir = os.path.join(SPLIT_ROOT, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    segment_samples, bad_files, skipped_not_target = [], [], 0
    for file_name in sorted(os.listdir(split_dir)):
        if not file_name.endswith(".npz"):
            continue
        m = pattern.search(file_name)
        if not m:
            bad_files.append((file_name, "filename pattern not matched")); continue
        exercise, person, trial, camera, seg_id = m.groups()
        if exercise != TARGET_EXERCISE:
            skipped_not_target += 1; continue
        key = (exercise, person, trial)
        if key not in score_map:
            bad_files.append((file_name, "missing label")); continue
        file_path = os.path.join(split_dir, file_name)
        try:
            kp = np.load(file_path)["keypoints_3d"]
        except Exception as e:
            bad_files.append((file_name, str(e))); continue
        if kp.ndim != 3 or kp.shape[1:] != (17, 3) or kp.shape[0] == 0:
            bad_files.append((file_name, f"bad shape {kp.shape}")); continue
        segment_samples.append({"file_name": file_name, "exercise": exercise, "person": person, "trial": trial, "camera": camera, "segment": int(seg_id), "x": kp.astype(np.float32), "y": score_map[key]})

    grouped = defaultdict(list)
    for s in segment_samples:
        grouped[(s["exercise"], s["person"], s["trial"], s["camera"])].append((s["segment"], s["x"], s["y"], s["file_name"]))

    trial_camera_sequences = defaultdict(dict)
    for key, segs in grouped.items():
        exercise, person, trial, camera = key
        segs = sorted(segs, key=lambda x: x[0])
        arrays = [arr for _, arr, _, _ in segs if arr.ndim == 3 and arr.shape[1:] == (17, 3) and arr.shape[0] > 0]
        if not arrays:
            bad_files.append((str(key), "no valid arrays after filtering")); continue
        try:
            full = np.concatenate(arrays, axis=0)
        except Exception as e:
            bad_files.append((str(key), f"concatenate error: {e}")); continue
        trial_camera_sequences[(exercise, person, trial)][camera] = {"x": full.astype(np.float32), "y": float(segs[0][2]), "file_name": segs[0][3], "frames": int(full.shape[0])}

    multiview_samples, incomplete_trials, label_rows, align_rows = [], [], [], []
    for trial_key, cam_dict in trial_camera_sequences.items():
        exercise, person, trial = trial_key
        available = set(cam_dict.keys())
        if not REQUIRED_CAMERAS.issubset(available):
            incomplete_trials.append({"split": split_name, "exercise": exercise, "person": person, "trial": trial, "available_cameras": ",".join(sorted(available)), "missing_cameras": ",".join(sorted(REQUIRED_CAMERAS - available))})
            continue
        y0, y1, y2 = cam_dict["C0"]["y"], cam_dict["C1"]["y"], cam_dict["C2"]["y"]
        ok = (y0 == y1 == y2)
        label_rows.append({"split": split_name, "exercise": exercise, "person": person, "trial": trial, "c0_label": y0, "c1_label": y1, "c2_label": y2, "label_match": ok})
        align_rows.append({"split": split_name, "exercise": exercise, "person": person, "trial": trial, "c0_file": cam_dict["C0"]["file_name"], "c1_file": cam_dict["C1"]["file_name"], "c2_file": cam_dict["C2"]["file_name"], "c0_frames_before_resample": cam_dict["C0"]["frames"], "c1_frames_before_resample": cam_dict["C1"]["frames"], "c2_frames_before_resample": cam_dict["C2"]["frames"], "target_len_after_resample": TARGET_LEN, "label": y0, "label_match": ok})
        multiview_samples.append({"split": split_name, "exercise": exercise, "person": person, "trial": trial, "x_c0": cam_dict["C0"]["x"], "x_c1": cam_dict["C1"]["x"], "x_c2": cam_dict["C2"]["x"], "y": y0})
    return multiview_samples, bad_files, incomplete_trials, skipped_not_target, label_rows, align_rows


def samples_to_df(samples, split_name):
    return pd.DataFrame([{"split": split_name, "exercise": s["exercise"], "person": s["person"], "trial": s["trial"], "score": s["y"], "performance": "good" if s["y"] >= 4.0 else "bad"} for s in samples])


def prepare_xy(samples):
    X0, X1, X2, y, meta = [], [], [], [], []
    for s in samples:
        X0.append(preprocess_one_view(s["x_c0"])); X1.append(preprocess_one_view(s["x_c1"])); X2.append(preprocess_one_view(s["x_c2"]))
        y.append(float(s["y"]))
        meta.append({"split": s["split"], "exercise": s["exercise"], "person": s["person"], "trial": s["trial"]})
    return np.array(X0, dtype=np.float32), np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32), np.array(y, dtype=np.float32), meta


def calculate_metrics(y_true, y_pred):
    pcc, pval = pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2_score(y_true, y_pred), "PCC": pcc, "PCC_p_value": pval}


def make_predictions_df(meta, y_true, y_pred):
    return pd.DataFrame([{"split": m["split"], "exercise": m["exercise"], "person": m["person"], "trial": m["trial"], "true_score": float(t), "predicted_score": float(p), "abs_error": float(abs(t-p))} for m, t, p in zip(meta, y_true, y_pred)])


def per_person_scores(df):
    rows = []
    for person, g in df.groupby("person"):
        true, pred = g["true_score"].values, g["predicted_score"].values
        rows.append({"person": person, "num_trials": len(g), "mean_true_score": np.mean(true), "mean_predicted_score": np.mean(pred), "MAE": mean_absolute_error(true, pred), "RMSE": np.sqrt(mean_squared_error(true, pred)), "R2": r2_score(true, pred) if len(g) > 1 else np.nan, "PCC": pearsonr(true, pred)[0] if len(g) > 1 else np.nan})
    return pd.DataFrame(rows).sort_values("person")


def get_ylim(values, pad_ratio=0.15):
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return -1, 1
    mn, mx = min(clean), max(clean)
    pad = (mx - mn) * pad_ratio if mx != mn else 0.1
    return mn - pad, mx + pad


fit_samples, bad_train, incomplete_train, skipped_train, label_train, align_train = load_multiview_split("train")
val_samples, bad_valid, incomplete_valid, skipped_valid, label_valid, align_valid = load_multiview_split("valid")
test_samples, bad_test, incomplete_test, skipped_test, label_test, align_test = load_multiview_split("test")

bad_files = bad_train + bad_valid + bad_test
incomplete_trials = incomplete_train + incomplete_valid + incomplete_test
label_check_rows = label_train + label_valid + label_test
camera_alignment_rows = align_train + align_valid + align_test

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w", encoding="utf-8") as f:
    for item in bad_files:
        f.write(str(item) + "\n")
pd.DataFrame(incomplete_trials).to_csv(os.path.join(OUTPUT_DIR, "incomplete_camera_trials.csv"), index=False)
pd.DataFrame(label_check_rows).to_csv(os.path.join(OUTPUT_DIR, "label_mismatch_check.csv"), index=False)
pd.DataFrame(camera_alignment_rows).to_csv(os.path.join(OUTPUT_DIR, "camera_alignment_check.csv"), index=False)

print("Fit samples:", len(fit_samples)); print("Validation samples:", len(val_samples)); print("Test samples:", len(test_samples))
print("Bad files:", len(bad_files)); print("Incomplete camera trials:", len(incomplete_trials)); print("Label mismatches:", sum(1 for r in label_check_rows if not r["label_match"]))

if len(fit_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
    raise ValueError("One split has zero samples. Check complete-camera availability.")

split_distribution_df = pd.concat([samples_to_df(fit_samples, "train"), samples_to_df(val_samples, "valid"), samples_to_df(test_samples, "test")], ignore_index=True)
split_distribution_df.to_csv(os.path.join(OUTPUT_DIR, "split_distribution.csv"), index=False)
person_summary_df = split_distribution_df.groupby(["split", "person"]).agg(num_samples=("score", "count"), mean_score=("score", "mean"), min_score=("score", "min"), max_score=("score", "max"), good_count=("performance", lambda x: (x == "good").sum()), bad_count=("performance", lambda x: (x == "bad").sum())).reset_index()
person_summary_df.to_csv(os.path.join(OUTPUT_DIR, "person_split_summary.csv"), index=False)

X0_fit, X1_fit, X2_fit, y_fit, fit_meta = prepare_xy(fit_samples)
X0_val, X1_val, X2_val, y_val, val_meta = prepare_xy(val_samples)
X0_test, X1_test, X2_test, y_test, test_meta = prepare_xy(test_samples)
print("X0_fit shape:", X0_fit.shape); print("X1_fit shape:", X1_fit.shape); print("X2_fit shape:", X2_fit.shape)
print("X0_val shape:", X0_val.shape); print("X0_test shape:", X0_test.shape)

y_scaler = StandardScaler()
y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
feature_dim = X0_fit.shape[2]


def build_branch(name):
    inp = Input(shape=(TARGET_LEN, feature_dim), name=name)
    x = Conv1D(16, 5, activation="relu", padding="same")(inp)
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
    return inp, x
input_c0, feat_c0 = build_branch("input_c0")
input_c1, feat_c1 = build_branch("input_c1")
input_c2, feat_c2 = build_branch("input_c2")
merged = Concatenate(name="late_fusion_concat")([feat_c0, feat_c1, feat_c2])
x = Dense(64, activation="relu")(merged)
x = Dropout(0.10)(x)
x = Dense(32, activation="relu")(x)
output = Dense(1)(x)
model = Model(inputs=[input_c0, input_c1, input_c2], outputs=output)


model.compile(optimizer=Adam(learning_rate=2e-5), loss="mse", metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
model.summary()

class RegressionMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train_list, y_train_scaled, X_val_list, y_val_scaled):
        super().__init__(); self.X_train_list = X_train_list; self.y_train_scaled = y_train_scaled; self.X_val_list = X_val_list; self.y_val_scaled = y_val_scaled
        self.train_r2, self.val_r2, self.train_pcc, self.val_pcc = [], [], [], []
    def on_epoch_end(self, epoch, logs=None):
        tr = self.model.predict(self.X_train_list, verbose=0).flatten(); va = self.model.predict(self.X_val_list, verbose=0).flatten()
        self.train_r2.append(r2_score(self.y_train_scaled, tr)); self.val_r2.append(r2_score(self.y_val_scaled, va))
        self.train_pcc.append(pearsonr(self.y_train_scaled, tr)[0] if len(self.y_train_scaled) > 1 else np.nan)
        self.val_pcc.append(pearsonr(self.y_val_scaled, va)[0] if len(self.y_val_scaled) > 1 else np.nan)

epoch_metrics = RegressionMetricsCallback([X0_fit, X1_fit, X2_fit], y_fit_scaled, [X0_val, X1_val, X2_val], y_val_scaled)
callbacks = [epoch_metrics, EarlyStopping(monitor="val_loss", patience=80, min_delta=0.00005, restore_best_weights=True, verbose=1), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1), ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1)]
history = model.fit([X0_fit, X1_fit, X2_fit], y_fit_scaled, validation_data=([X0_val, X1_val, X2_val], y_val_scaled), epochs=EPOCHS, batch_size=256, callbacks=callbacks, shuffle=True, verbose=1)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    for i in range(len(history.history["loss"])):
        f.write(f"Epoch {i+1}\nTrain Loss = {history.history['loss'][i]}\nVal Loss = {history.history['val_loss'][i]}\nTrain MAE = {history.history['mae'][i]}\nVal MAE = {history.history['val_mae'][i]}\nTrain RMSE = {history.history['rmse'][i]}\nVal RMSE = {history.history['val_rmse'][i]}\nTrain R2 = {epoch_metrics.train_r2[i]}\nVal R2 = {epoch_metrics.val_r2[i]}\nTrain PCC = {epoch_metrics.train_pcc[i]}\nVal PCC = {epoch_metrics.val_pcc[i]}\n\n")

train_pred_scaled = model.predict([X0_fit, X1_fit, X2_fit], verbose=0).flatten()
val_pred_scaled = model.predict([X0_val, X1_val, X2_val], verbose=0).flatten()
test_pred_scaled = model.predict([X0_test, X1_test, X2_test], verbose=0).flatten()
train_loss, train_mae_scaled, train_rmse_scaled = model.evaluate([X0_fit, X1_fit, X2_fit], y_fit_scaled, verbose=0)
val_loss, val_mae_scaled, val_rmse_scaled = model.evaluate([X0_val, X1_val, X2_val], y_val_scaled, verbose=0)
test_loss, test_mae_scaled, test_rmse_scaled = model.evaluate([X0_test, X1_test, X2_test], y_test_scaled, verbose=0)
train_pred = np.clip(y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten(), 1.0, 5.0)
val_pred = np.clip(y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten(), 1.0, 5.0)
test_pred = np.clip(y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten(), 1.0, 5.0)
train_metrics, val_metrics, test_metrics = calculate_metrics(y_fit, train_pred), calculate_metrics(y_val, val_pred), calculate_metrics(y_test, test_pred)

print("\n===== TRAIN RESULTS ====="); [print(k, "=", v) for k, v in train_metrics.items()]
print("\n===== VALIDATION RESULTS ====="); [print(k, "=", v) for k, v in val_metrics.items()]
print("\n===== TEST RESULTS ====="); [print(k, "=", v) for k, v in test_metrics.items()]

train_predictions_df = make_predictions_df(fit_meta, y_fit, train_pred)
validation_predictions_df = make_predictions_df(val_meta, y_val, val_pred)
test_predictions_df = make_predictions_df(test_meta, y_test, test_pred)
train_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)
validation_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "validation_predictions.csv"), index=False)
test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
all_split_predictions_df = pd.concat([train_predictions_df, validation_predictions_df, test_predictions_df], ignore_index=True)
all_split_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "all_split_predictions.csv"), index=False)

train_pp, val_pp, test_pp = per_person_scores(train_predictions_df), per_person_scores(validation_predictions_df), per_person_scores(test_predictions_df)
train_pp.to_csv(os.path.join(OUTPUT_DIR, "train_per_person_scores.csv"), index=False)
val_pp.to_csv(os.path.join(OUTPUT_DIR, "validation_per_person_scores.csv"), index=False)
test_pp.to_csv(os.path.join(OUTPUT_DIR, "per_person_scores.csv"), index=False)
worst_predictions_df = test_predictions_df.sort_values("abs_error", ascending=False).head(30)
worst_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "worst_30_predictions.csv"), index=False)

pd.DataFrame([
    {"split": "train", "loss_scaled": train_loss, "mae_scaled": train_mae_scaled, "rmse_scaled": train_rmse_scaled, **train_metrics},
    {"split": "validation", "loss_scaled": val_loss, "mae_scaled": val_mae_scaled, "rmse_scaled": val_rmse_scaled, **val_metrics},
    {"split": "test", "loss_scaled": test_loss, "mae_scaled": test_mae_scaled, "rmse_scaled": test_rmse_scaled, **test_metrics},
]).to_csv(os.path.join(OUTPUT_DIR, "train_validation_test_metrics.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"===== Multi-view {TARGET_EXERCISE} {FUSION_TITLE} Velocity Only Results =====\n")
    f.write(f"Output folder = {OUTPUT_DIR}\nBatch name = {args.batch_name}\nTarget_len = {TARGET_LEN}\nEpochs requested = {EPOCHS}\nEpochs trained = {len(history.history['loss'])}\n\n")
    f.write("===== Train Metrics =====\n"); [f.write(f"{k} = {v}\n") for k, v in train_metrics.items()]; f.write(f"Train Loss scaled = {train_loss}\nTrain MAE scaled = {train_mae_scaled}\nTrain RMSE scaled = {train_rmse_scaled}\n\n")
    f.write("===== Validation Metrics =====\n"); [f.write(f"{k} = {v}\n") for k, v in val_metrics.items()]; f.write(f"Validation Loss scaled = {val_loss}\nValidation MAE scaled = {val_mae_scaled}\nValidation RMSE scaled = {val_rmse_scaled}\n\n")
    f.write("===== Test Metrics =====\n"); [f.write(f"{k} = {v}\n") for k, v in test_metrics.items()]; f.write(f"Test Loss scaled = {test_loss}\nTest MAE scaled = {test_mae_scaled}\nTest RMSE scaled = {test_rmse_scaled}\n\n")
    f.write("===== Dataset Sizes =====\n")
    f.write(f"Fit samples = {len(fit_samples)}\nValidation samples = {len(val_samples)}\nTest samples = {len(test_samples)}\nX0_fit shape = {X0_fit.shape}\nX1_fit shape = {X1_fit.shape}\nX2_fit shape = {X2_fit.shape}\nBad skipped files = {len(bad_files)}\nIncomplete camera trials = {len(incomplete_trials)}\nSkipped non-target files = {skipped_train + skipped_valid + skipped_test}\nLabel mismatch rows = {sum(1 for r in label_check_rows if not r['label_match'])}\n\n")
    f.write("===== Split Person Summary =====\n" + person_summary_df.to_string(index=False) + "\n\n")
    f.write("===== Train Per-person Scores =====\n" + train_pp.to_string(index=False) + "\n\n")
    f.write("===== Validation Per-person Scores =====\n" + val_pp.to_string(index=False) + "\n\n")
    f.write("===== Test Per-person Scores =====\n" + test_pp.to_string(index=False) + "\n\n")
    f.write("===== Worst 30 Test Predictions =====\n" + worst_predictions_df.to_string(index=False))

np.savez(os.path.join(OUTPUT_DIR, "plot_data.npz"), train_loss=np.array(history.history["loss"]), val_loss=np.array(history.history["val_loss"]), train_mae=np.array(history.history["mae"]), val_mae=np.array(history.history["val_mae"]), train_rmse=np.array(history.history["rmse"]), val_rmse=np.array(history.history["val_rmse"]), train_r2=np.array(epoch_metrics.train_r2), val_r2=np.array(epoch_metrics.val_r2), y_train=y_fit, y_train_pred=train_pred, y_val=y_val, y_val_pred=val_pred, y_test=y_test, y_test_pred=test_pred)

loss_ymin, loss_ymax = get_ylim(history.history["loss"] + history.history["val_loss"] + [test_loss])
plt.figure(figsize=(9,5)); plt.plot(history.history["loss"], label="Train", linewidth=2); plt.plot(history.history["val_loss"], label="Validation", linewidth=2); plt.axhline(y=test_loss, linestyle="-.", linewidth=2, label=f"Test Loss={test_loss:.4f}"); plt.title(f"{TARGET_EXERCISE} {FUSION_TITLE} Loss (MSE)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.ylim(loss_ymin, loss_ymax); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight"); plt.close()
rmse_ymin, rmse_ymax = get_ylim(history.history["rmse"] + history.history["val_rmse"] + [test_rmse_scaled])
mae_ymin, mae_ymax = get_ylim(history.history["mae"] + history.history["val_mae"] + [test_mae_scaled])
plt.figure(figsize=(12,5)); plt.subplot(1,2,1); plt.plot(history.history["rmse"], label="Train", linewidth=2); plt.plot(history.history["val_rmse"], label="Validation", linewidth=2); plt.axhline(y=test_rmse_scaled, linestyle="-.", linewidth=2, label=f"Test RMSE={test_rmse_scaled:.4f}"); plt.title("RMSE"); plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.ylim(rmse_ymin, rmse_ymax); plt.grid(True, alpha=0.25); plt.legend(); plt.subplot(1,2,2); plt.plot(history.history["mae"], label="Train", linewidth=2); plt.plot(history.history["val_mae"], label="Validation", linewidth=2); plt.axhline(y=test_mae_scaled, linestyle="-.", linewidth=2, label=f"Test MAE={test_mae_scaled:.4f}"); plt.title("MAE"); plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.ylim(mae_ymin, mae_ymax); plt.grid(True, alpha=0.25); plt.legend(); plt.suptitle(f"{TARGET_EXERCISE} {FUSION_TITLE} RMSE & MAE"); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "rmse_mae.png"), dpi=300, bbox_inches="tight"); plt.close()
r2_ymin, r2_ymax = get_ylim(epoch_metrics.train_r2 + epoch_metrics.val_r2 + [test_metrics["R2"], 0, 1], 0.10)
plt.figure(figsize=(9,5)); plt.plot(epoch_metrics.train_r2, label="Train", linewidth=2); plt.plot(epoch_metrics.val_r2, label="Validation", linewidth=2); plt.axhline(y=test_metrics["R2"], linestyle="-.", linewidth=2, label=f"Test R2={test_metrics['R2']:.4f}"); plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect"); plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="Baseline"); plt.title(f"{TARGET_EXERCISE} {FUSION_TITLE} R²"); plt.xlabel("Epoch"); plt.ylabel("R²"); plt.ylim(r2_ymin, r2_ymax); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "r2_curve.png"), dpi=300, bbox_inches="tight"); plt.close()
pcc_ymin, pcc_ymax = get_ylim(epoch_metrics.train_pcc + epoch_metrics.val_pcc + [test_metrics["PCC"], 0, 1], 0.10)
plt.figure(figsize=(9,5)); plt.plot(epoch_metrics.train_pcc, label="Train", linewidth=2); plt.plot(epoch_metrics.val_pcc, label="Validation", linewidth=2); plt.axhline(y=test_metrics["PCC"], linestyle="-.", linewidth=2, label=f"Test PCC={test_metrics['PCC']:.4f}"); plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect"); plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="No correlation"); plt.title(f"{TARGET_EXERCISE} {FUSION_TITLE} PCC"); plt.xlabel("Epoch"); plt.ylabel("PCC"); plt.ylim(pcc_ymin, pcc_ymax); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "pcc_curve.png"), dpi=300, bbox_inches="tight"); plt.close()
plt.figure(figsize=(6,6)); plt.scatter(y_test, test_pred, alpha=0.7); plt.plot([1,5], [1,5], "r--"); plt.xlim(1,5); plt.ylim(1,5); plt.xlabel("True Score"); plt.ylabel("Predicted Score"); plt.title(f"{TARGET_EXERCISE} {FUSION_TITLE} True vs Predicted"); plt.grid(True, alpha=0.25); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight"); plt.close()
image_files = ["loss_curve.png", "rmse_mae.png", "r2_curve.png", "pcc_curve.png", "true_vs_predicted.png"]
with tarfile.open(os.path.join(OUTPUT_DIR, "images.tar.gz"), "w:gz") as tar:
    for img in image_files:
        img_path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(img_path): tar.add(img_path, arcname=img)
print("\nSaved files in:", OUTPUT_DIR)
