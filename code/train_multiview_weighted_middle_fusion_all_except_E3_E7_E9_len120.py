
import os, re, random, argparse, tarfile
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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Concatenate, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument("--target_len", type=int, default=120)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

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

EXCLUDED_EXERCISES = {"E3", "E7", "E9"}
REQUIRED_CAMERAS = {"C0", "C1", "C2"}

# Fixed weights. Change if your single-camera results suggest different values.
W_C0 = 0.50
W_C1 = 0.25
W_C2 = 0.25

OUTPUT_DIR = f"/mvdlph/shahd/MVCNNPH/results_multiview_weighted_middle_fusion_all_except_E3_E7_E9_velocity_only/len{TARGET_LEN}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Experiment: Multi-view Weighted Middle Fusion Velocity Only")
print("Included exercises: all except", sorted(EXCLUDED_EXERCISES))
print("Weights: C0=", W_C0, "C1=", W_C1, "C2=", W_C2)
print("Target length:", TARGET_LEN)
print("Output:", OUTPUT_DIR)

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    score_map[(row["exercise"], row["person"], row["trial"])] = float(row["mean"])

pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")


def safe_pcc(true, pred):
    if len(true) <= 1:
        return np.nan, np.nan
    try:
        return pearsonr(true, pred)
    except Exception:
        return np.nan, np.nan


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
    skipped_excluded = 0

    for file_name in sorted(os.listdir(split_dir)):
        if not file_name.endswith(".npz"):
            continue
        match = pattern.search(file_name)
        if not match:
            bad_files.append((file_name, "filename pattern not matched"))
            continue

        exercise, person, trial, camera, seg_id = match.groups()

        if exercise in EXCLUDED_EXERCISES:
            skipped_excluded += 1
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

        segment_samples.append({
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "segment": int(seg_id),
            "x": keypoints_3d.astype(np.float32),
            "y": score_map[label_key]
        })

    grouped = defaultdict(list)
    for sample in segment_samples:
        key = (sample["exercise"], sample["person"], sample["trial"], sample["camera"])
        grouped[key].append((sample["segment"], sample["x"], sample["y"]))

    trial_camera_sequences = defaultdict(dict)

    for key, segs in grouped.items():
        exercise, person, trial, camera = key
        segs = sorted(segs, key=lambda item: item[0])
        arrays = []
        for _, arr, _ in segs:
            if arr.ndim == 3 and arr.shape[1:] == (17, 3) and arr.shape[0] > 0:
                arrays.append(arr)
        if len(arrays) == 0:
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
            "y": float(segs[0][2])
        }

    multiview_samples = []
    incomplete_trials = []

    for trial_key, cam_dict in trial_camera_sequences.items():
        available = set(cam_dict.keys())
        if not REQUIRED_CAMERAS.issubset(available):
            incomplete_trials.append({
                "split": split_name,
                "exercise": trial_key[0],
                "person": trial_key[1],
                "trial": trial_key[2],
                "available": ",".join(sorted(available)),
                "missing": ",".join(sorted(REQUIRED_CAMERAS - available))
            })
            continue

        multiview_samples.append({
            "exercise": trial_key[0],
            "person": trial_key[1],
            "trial": trial_key[2],
            "x_c0": cam_dict["C0"]["x"],
            "x_c1": cam_dict["C1"]["x"],
            "x_c2": cam_dict["C2"]["x"],
            "y": cam_dict["C0"]["y"]
        })

    return multiview_samples, bad_files, incomplete_trials, skipped_excluded


fit_samples, bad_train, incomplete_train, skipped_train = load_multiview_split("train")
val_samples, bad_valid, incomplete_valid, skipped_valid = load_multiview_split("valid")
test_samples, bad_test, incomplete_test, skipped_test = load_multiview_split("test")

bad_files = bad_train + bad_valid + bad_test
incomplete_trials = incomplete_train + incomplete_valid + incomplete_test

with open(os.path.join(OUTPUT_DIR, "bad_files.txt"), "w", encoding="utf-8") as f:
    for item in bad_files:
        f.write(str(item) + "\n")

pd.DataFrame(incomplete_trials).to_csv(os.path.join(OUTPUT_DIR, "incomplete_camera_trials.csv"), index=False)

print("Train:", len(fit_samples))
print("Validation:", len(val_samples))
print("Test:", len(test_samples))
print("Bad files:", len(bad_files))
print("Incomplete trials:", len(incomplete_trials))
print("Skipped excluded exercises:", skipped_train + skipped_valid + skipped_test)

if len(fit_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
    raise ValueError("One split has zero samples. Check data and excluded exercises.")


def samples_to_df(samples, split_name):
    rows = []
    for s in samples:
        rows.append({
            "split": split_name,
            "exercise": s["exercise"],
            "person": s["person"],
            "trial": s["trial"],
            "score": s["y"],
            "performance": "good" if s["y"] >= 4.0 else "bad"
        })
    return pd.DataFrame(rows)


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
    bad_count=("performance", lambda x: (x == "bad").sum())
).reset_index()
person_summary_df.to_csv(os.path.join(OUTPUT_DIR, "person_split_summary.csv"), index=False)

exercise_summary_df = split_distribution_df.groupby(["split", "exercise"]).agg(
    num_samples=("score", "count"),
    mean_score=("score", "mean"),
    min_score=("score", "min"),
    max_score=("score", "max"),
    good_count=("performance", lambda x: (x == "good").sum()),
    bad_count=("performance", lambda x: (x == "bad").sum())
).reset_index()
exercise_summary_df.to_csv(os.path.join(OUTPUT_DIR, "exercise_split_summary.csv"), index=False)


def prepare_xy(sample_list):
    X0, X1, X2, y, meta = [], [], [], [], []
    for sample in sample_list:
        c0 = preprocess_one_view(sample["x_c0"])
        c1 = preprocess_one_view(sample["x_c1"])
        c2 = preprocess_one_view(sample["x_c2"])
        X0.append(c0)
        X1.append(c1)
        X2.append(c2)
        y.append(float(sample["y"]))
        meta.append({
            "exercise": sample["exercise"],
            "person": sample["person"],
            "trial": sample["trial"]
        })
    return np.array(X0, dtype=np.float32), np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32), np.array(y, dtype=np.float32), meta


X0_fit, X1_fit, X2_fit, y_fit, fit_meta = prepare_xy(fit_samples)
X0_val, X1_val, X2_val, y_val, val_meta = prepare_xy(val_samples)
X0_test, X1_test, X2_test, y_test, test_meta = prepare_xy(test_samples)

print("X0_fit:", X0_fit.shape)
print("X1_fit:", X1_fit.shape)
print("X2_fit:", X2_fit.shape)
print("X0_val:", X0_val.shape)
print("X0_test:", X0_test.shape)

y_scaler = StandardScaler()
y_fit_scaled = y_scaler.fit_transform(y_fit.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

feature_dim = X0_fit.shape[2]


def build_branch(input_name):
    inp = Input(shape=(TARGET_LEN, feature_dim), name=input_name)
    x = Conv1D(16, 5, activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(24, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    return inp, x


input_c0, feat_c0 = build_branch("input_c0")
input_c1, feat_c1 = build_branch("input_c1")
input_c2, feat_c2 = build_branch("input_c2")

weighted_c0 = Lambda(lambda x: x * W_C0, name="weighted_c0")(feat_c0)
weighted_c1 = Lambda(lambda x: x * W_C1, name="weighted_c1")(feat_c1)
weighted_c2 = Lambda(lambda x: x * W_C2, name="weighted_c2")(feat_c2)

weighted_middle_fusion = Concatenate(axis=-1, name="weighted_middle_concat")([weighted_c0, weighted_c1, weighted_c2])

x = Conv1D(32, 3, activation="relu", padding="same")(weighted_middle_fusion)
x = BatchNormalization()(x)
x = Conv1D(48, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.10)(x)
x = Dense(32, activation="relu")(x)
output = Dense(1)(x)

model = Model(inputs=[input_c0, input_c1, input_c2], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)
model.summary()


class RegressionMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_r2 = []
        self.val_r2 = []
        self.train_pcc = []
        self.val_pcc = []

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(self.X_train, verbose=0).flatten()
        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        train_r2 = r2_score(self.y_train, train_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        train_pcc, _ = safe_pcc(self.y_train, train_pred)
        val_pcc, _ = safe_pcc(self.y_val, val_pred)
        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)
        self.train_pcc.append(train_pcc)
        self.val_pcc.append(val_pcc)


epoch_metrics = RegressionMetricsCallback([X0_fit, X1_fit, X2_fit], y_fit_scaled, [X0_val, X1_val, X2_val], y_val_scaled)

checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
callbacks = [
    epoch_metrics,
    EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    [X0_fit, X1_fit, X2_fit],
    y_fit_scaled,
    validation_data=([X0_val, X1_val, X2_val], y_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
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

y_pred_scaled = model.predict([X0_test, X1_test, X2_test]).flatten()
test_loss, test_mae_scaled, test_rmse_scaled = model.evaluate([X0_test, X1_test, X2_test], y_test_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.clip(y_pred, 1.0, 5.0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
pcc, pcc_pvalue = safe_pcc(y_test, y_pred)

print("\n===== Overall Test Results =====")
print("MAE =", mae)
print("RMSE =", rmse)
print("R2 =", r2)
print("PCC =", pcc)
print("PCC p-value =", pcc_pvalue)
print("Test Loss =", test_loss)
print("Test MAE scaled =", test_mae_scaled)
print("Test RMSE scaled =", test_rmse_scaled)

predictions_df = pd.DataFrame({
    "exercise": [m["exercise"] for m in test_meta],
    "person": [m["person"] for m in test_meta],
    "trial": [m["trial"] for m in test_meta],
    "true_score": y_test,
    "predicted_score": y_pred,
    "abs_error": np.abs(y_test - y_pred)
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
worst_predictions_df = predictions_df.sort_values("abs_error", ascending=False).head(30)
worst_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "worst_30_predictions.csv"), index=False)

per_exercise_rows = []
for exercise, group in predictions_df.groupby("exercise"):
    true = group["true_score"].values
    pred = group["predicted_score"].values
    ex_pcc, _ = safe_pcc(true, pred)
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
    person_pcc, _ = safe_pcc(true, pred)
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
    f.write("===== Multi-view Weighted Middle Fusion Velocity Only: All Exercises Except E3, E7, E9 =====\n\n")
    f.write(f"Target_len = {TARGET_LEN}\n")
    f.write(f"Epochs requested = {EPOCHS}\n")
    f.write(f"Epochs trained = {len(history.history['loss'])}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write("Fusion type = Weighted middle fusion\n")
    f.write("Fusion method = multiply camera feature maps by fixed weights, then concatenate after two CNN blocks per camera\n")
    f.write(f"Camera weights: C0={W_C0}, C1={W_C1}, C2={W_C2}\n")
    f.write("Views = C0 + C1 + C2\n")
    f.write("Complete-camera rule = if any camera is missing, the full trial is skipped\n")
    f.write("Included exercises = all except E3, E7, E9\n")
    f.write(f"Excluded exercises = {sorted(EXCLUDED_EXERCISES)}\n")
    f.write("Added features = velocity only\n")
    f.write("Loss = MSE\n")
    f.write("Learning rate = 2e-5\n")
    f.write("Branch filters = 16, 24\n")
    f.write("Shared filters after fusion = 32, 48\n")
    f.write("Dense layers = 64, 32\n")
    f.write("Dropout = 0.10\n")
    f.write("Images archive = images.tar.gz\n\n")
    f.write("===== Overall Test Metrics =====\n")
    f.write(f"MAE = {mae}\n")
    f.write(f"RMSE = {rmse}\n")
    f.write(f"R2 = {r2}\n")
    f.write(f"PCC = {pcc}\n")
    f.write(f"PCC p-value = {pcc_pvalue}\n")
    f.write(f"Test Loss = {test_loss}\n")
    f.write(f"Test MAE scaled = {test_mae_scaled}\n")
    f.write(f"Test RMSE scaled = {test_rmse_scaled}\n\n")
    f.write("===== Dataset Sizes =====\n")
    f.write(f"Train samples = {len(fit_samples)}\n")
    f.write(f"Validation samples = {len(val_samples)}\n")
    f.write(f"Test samples = {len(test_samples)}\n")
    f.write(f"X0_fit shape = {X0_fit.shape}\n")
    f.write(f"X1_fit shape = {X1_fit.shape}\n")
    f.write(f"X2_fit shape = {X2_fit.shape}\n")
    f.write(f"Incomplete trials removed = {len(incomplete_trials)}\n")
    f.write(f"Bad files = {len(bad_files)}\n")
    f.write(f"Skipped excluded exercise files = {skipped_train + skipped_valid + skipped_test}\n\n")
    f.write("===== Split Person Summary =====\n")
    f.write(person_summary_df.to_string(index=False))
    f.write("\n\n===== Split Exercise Summary =====\n")
    f.write(exercise_summary_df.to_string(index=False))
    f.write("\n\n===== Per-exercise scores =====\n")
    f.write(per_exercise_df.to_string(index=False))
    f.write("\n\n===== Per-person scores =====\n")
    f.write(per_person_df.to_string(index=False))
    f.write("\n\n===== Worst 30 predictions =====\n")
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
    pad = (vmax - vmin) * pad_ratio if vmax != vmin else 0.1
    return vmin - pad, vmax + pad


loss_values = history.history["loss"]
val_loss_values = history.history["val_loss"]
all_loss = loss_values + val_loss_values + [test_loss]
loss_ymin, loss_ymax = get_ylim(all_loss)

plt.figure(figsize=(9, 5))
plt.plot(loss_values, label="Train", linewidth=2)
plt.plot(val_loss_values, label="Validation", linewidth=2)
plt.axhline(y=test_loss, linestyle="-.", linewidth=2, label=f"Test={test_loss:.4f}")
plt.title("Weighted Middle Fusion Loss")
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
plt.suptitle("Weighted Middle Fusion RMSE & MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_mae.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([1, 5], [1, 5], "r--")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("True vs Predicted")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

all_r2 = epoch_metrics.train_r2 + epoch_metrics.val_r2 + [r2, 0, 1]
r2_ymin, r2_ymax = get_ylim(all_r2, pad_ratio=0.10)
plt.figure(figsize=(9, 5))
plt.plot(epoch_metrics.train_r2, label="Train", linewidth=2)
plt.plot(epoch_metrics.val_r2, label="Validation", linewidth=2)
plt.axhline(y=r2, linestyle="-.", linewidth=2, label=f"Test R2={r2:.4f}")
plt.axhline(y=1.0, linestyle=":", linewidth=1.5, label="Perfect")
plt.axhline(y=0.0, linestyle=":", linewidth=1.5, label="Baseline")
plt.title("R2")
plt.xlabel("Epoch")
plt.ylabel("R2")
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
plt.title("PCC")
plt.xlabel("Epoch")
plt.ylabel("PCC")
plt.ylim(pcc_ymin, pcc_ymax)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pcc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

image_files = ["loss_curve.png", "rmse_mae.png", "true_vs_predicted.png", "r2_curve.png", "pcc_curve.png"]
tar_path = os.path.join(OUTPUT_DIR, "images.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    for img in image_files:
        img_path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(img_path):
            tar.add(img_path, arcname=img)

print("Saved in:", OUTPUT_DIR)
print("metrics.txt")
print("training_log.txt")
print("predictions.csv")
print("worst_30_predictions.csv")
print("per_exercise_scores.csv")
print("per_person_scores.csv")
print("split_distribution.csv")
print("person_split_summary.csv")
print("exercise_split_summary.csv")
print("plot_data.npz")
print("loss_curve.png")
print("rmse_mae.png")
print("true_vs_predicted.png")
print("r2_curve.png")
print("pcc_curve.png")
print("images.tar.gz")
print("bad_files.txt")
print("incomplete_camera_trials.csv")
print("best_model.keras")
