import os
import pandas as pd

BASE_RESULTS = "/mvdlph/shahd/MVCNNPH/results/exercise_E0_E1_results"

rows = []

for exercise in ["E0", "E1"]:
    exercise_path = os.path.join(BASE_RESULTS, exercise)

    # Single view
    single_path = os.path.join(exercise_path, "single_view")
    if os.path.isdir(single_path):
        for cam in ["C0", "C1", "C2", "all_cameras"]:
            cam_path = os.path.join(single_path, cam)

            for root, dirs, files in os.walk(cam_path):
                if "train_validation_test_metrics.csv" in files:
                    metrics_path = os.path.join(root, "train_validation_test_metrics.csv")
                    df = pd.read_csv(metrics_path)

                    row = {
                        "Exercise": exercise,
                        "View": "Single View",
                        "Method": cam,
                        "Model": "CNN"
                    }

                    for _, r in df.iterrows():
                        split = r["split"].capitalize()
                        row[f"{split} MAE"] = r.get("MAE")
                        row[f"{split} RMSE"] = r.get("RMSE")
                        row[f"{split} R2"] = r.get("R2")
                        row[f"{split} PCC"] = r.get("PCC")

                    rows.append(row)

    # Multi view
    multi_path = os.path.join(exercise_path, "multiview")
    if os.path.isdir(multi_path):
        for method in ["early_fusion", "middle_fusion", "late_fusion"]:
            method_path = os.path.join(multi_path, method)

            for root, dirs, files in os.walk(method_path):
                if "train_validation_test_metrics.csv" in files:
                    metrics_path = os.path.join(root, "train_validation_test_metrics.csv")
                    df = pd.read_csv(metrics_path)

                    row = {
                        "Exercise": exercise,
                        "View": "Multi View",
                        "Method": method,
                        "Model": "CNN"
                    }

                    for _, r in df.iterrows():
                        split = r["split"].capitalize()
                        row[f"{split} MAE"] = r.get("MAE")
                        row[f"{split} RMSE"] = r.get("RMSE")
                        row[f"{split} R2"] = r.get("R2")
                        row[f"{split} PCC"] = r.get("PCC")

                    rows.append(row)

summary_df = pd.DataFrame(rows)

output_csv = "/mvdlph/shahd/MVCNNPH/results/exercise_E0_E1_results/all_results_summary.csv"

summary_df.to_csv(output_csv, index=False)

print("Saved:")
print(output_csv)
print(summary_df)
