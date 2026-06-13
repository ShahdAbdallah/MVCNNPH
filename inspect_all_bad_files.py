import os
import re
import numpy as np
import pandas as pd

DATASET_PATH = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
OUTPUT_CSV = "/mvdlph/shahd/MVCNNPH/all_bad_files_summary.csv"

file_pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")

bad_files = []
total_files = 0
valid_files = 0

for file_name in sorted(os.listdir(DATASET_PATH)):
    if not file_name.endswith(".npz"):
        continue

    total_files += 1
    file_path = os.path.join(DATASET_PATH, file_name)

    match = file_pattern.search(file_name)
    if match:
        exercise, person, trial, camera, segment = match.groups()
    else:
        exercise = person = trial = camera = segment = "UNKNOWN"

    try:
        data = np.load(file_path)

        if "keypoints_3d" not in data.files:
            bad_files.append({
                "file_name": file_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "camera": camera,
                "segment": segment,
                "problem": "missing keypoints_3d",
                "shape": "N/A"
            })
            continue

        arr = data["keypoints_3d"]

        if arr.ndim != 3 or arr.shape[1:] != (17, 3) or arr.shape[0] == 0:
            bad_files.append({
                "file_name": file_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "camera": camera,
                "segment": segment,
                "problem": "invalid or empty keypoints_3d",
                "shape": str(arr.shape)
            })
        else:
            valid_files += 1

    except Exception as e:
        bad_files.append({
            "file_name": file_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "segment": segment,
            "problem": f"load error: {e}",
            "shape": "N/A"
        })

bad_df = pd.DataFrame(bad_files)
bad_df.to_csv(OUTPUT_CSV, index=False)

print("====================================")
print("Bad Files Inspection Summary")
print("====================================")
print("Total .npz files:", total_files)
print("Valid files     :", valid_files)
print("Bad files       :", len(bad_files))
print("Saved CSV to    :", OUTPUT_CSV)

if len(bad_files) > 0:
    print("\nBad files by camera:")
    print(bad_df["camera"].value_counts())

    print("\nBad files by problem:")
    print(bad_df["problem"].value_counts())

    print("\nFirst bad files:")
    print(bad_df.head(20).to_string(index=False))
else:
    print("\nNo bad files found.")
import os
import re
import numpy as np
import pandas as pd

DATASET_PATH = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"
OUTPUT_CSV = "/mvdlph/shahd/MVCNNPH/all_bad_files_summary.csv"

file_pattern = re.compile(r"(E\d+)_(P\d+)_(T\d+)_(C\d+)_seg(\d+)")

bad_files = []
total_files = 0
valid_files = 0

for file_name in sorted(os.listdir(DATASET_PATH)):
    if not file_name.endswith(".npz"):
        continue

    total_files += 1
    file_path = os.path.join(DATASET_PATH, file_name)

    match = file_pattern.search(file_name)
    if match:
        exercise, person, trial, camera, segment = match.groups()
    else:
        exercise = person = trial = camera = segment = "UNKNOWN"

    try:
        data = np.load(file_path)

        if "keypoints_3d" not in data.files:
            bad_files.append({
                "file_name": file_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "camera": camera,
                "segment": segment,
                "problem": "missing keypoints_3d",
                "shape": "N/A"
            })
            continue

        arr = data["keypoints_3d"]

        if arr.ndim != 3 or arr.shape[1:] != (17, 3) or arr.shape[0] == 0:
            bad_files.append({
                "file_name": file_name,
                "exercise": exercise,
                "person": person,
                "trial": trial,
                "camera": camera,
                "segment": segment,
                "problem": "invalid or empty keypoints_3d",
                "shape": str(arr.shape)
            })
        else:
            valid_files += 1

    except Exception as e:
        bad_files.append({
            "file_name": file_name,
            "exercise": exercise,
            "person": person,
            "trial": trial,
            "camera": camera,
            "segment": segment,
            "problem": f"load error: {e}",
            "shape": "N/A"
        })

bad_df = pd.DataFrame(bad_files)
bad_df.to_csv(OUTPUT_CSV, index=False)

print("====================================")
print("Bad Files Inspection Summary")
print("====================================")
print("Total .npz files:", total_files)
print("Valid files     :", valid_files)
print("Bad files       :", len(bad_files))
print("Saved CSV to    :", OUTPUT_CSV)

if len(bad_files) > 0:
    print("\nBad files by camera:")
    print(bad_df["camera"].value_counts())

    print("\nBad files by problem:")
    print(bad_df["problem"].value_counts())

    print("\nFirst bad files:")
    print(bad_df.head(20).to_string(index=False))
else:
    print("\nNo bad files found.")
