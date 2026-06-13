import pandas as pd

CSV_PATH = "/mvdlph/label_events_20260129_155122_stats_short.csv"

df = pd.read_csv(CSV_PATH)
df["exercise"] = df["exercise"].astype(str).str.strip()

print("\n===== E0 Overall =====")
print(df[df["exercise"]=="E0"]["mean"].describe())

print("\n===== E3 Overall =====")
print(df[df["exercise"]=="E3"]["mean"].describe())

print("\n===== E0 by person =====")
print(df[df["exercise"]=="E0"].groupby("person")["mean"].describe())

print("\n===== E3 by person =====")
print(df[df["exercise"]=="E3"].groupby("person")["mean"].describe())
