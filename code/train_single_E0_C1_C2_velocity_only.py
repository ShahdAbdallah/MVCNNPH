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
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DATASET = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
SPLIT_ROOT = os.path.join(BASE_DATASET, "by_person")
CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

TARGET_EXERCISE = "E0"
TARGET_CAMERAS = ["C1", "C2"]

labels_df = pd.read_csv(CSV_PATH)
labels_df["exercise"] = labels_df["exercise"].astype(str).str.strip()
labels_df["person"] = labels_df["person"].astype(str).str.strip()
labels_df["trial"] = labels_df["trial"].astype(str).str.strip()

score_map = {}
for _, row in labels_df.iterrows():
    score_map[(row["exercise"], row["person"], row["trial"])] = float(row["mean"])

pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_name, target_camera):
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

        if exercise != TARGET_EXERCISE or camera != target_camera:
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

    return trial_samples, bad_files, skipped_not_target
