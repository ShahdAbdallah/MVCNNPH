import numpy as np

BASE = "/mvdlph/Dataset_CVDLPT_Videos_Segments_P0P15_MMPose_human3d_motionbert_H36M_3D_1_2026"

bad_file = "E0_P13_T5_C2_seg2_MMPose_human3d_motionbert_3D.npz"
good_file = "E0_P0_T0_C2_seg0_MMPose_human3d_motionbert_3D.npz"

def inspect(file):
    path = f"{BASE}/{file}"
    print("\n==============================")
    print("FILE:", file)

    try:
        data = np.load(path)
        print("Keys:", data.files)

        if "keypoints_3d" in data:
            arr = data["keypoints_3d"]
            print("Shape:", arr.shape)
            print("Dimensions:", arr.ndim)

            if arr.ndim == 3 and arr.shape[0] > 0:
                print("First frame:")
                print(arr[0])
            else:
                print("Empty or invalid array")

        else:
            print("No keypoints_3d key found")

    except Exception as e:
        print("Error loading file:", e)

inspect(bad_file)
inspect(good_file)

print("\n======= COMPARISON =======")

try:
    bad = np.load(f"{BASE}/{bad_file}")["keypoints_3d"]
    good = np.load(f"{BASE}/{good_file}")["keypoints_3d"]

    print("BAD shape :", bad.shape)
    print("GOOD shape:", good.shape)
except:
    pass


