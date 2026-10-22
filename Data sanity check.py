"""LiDAR data summary table + text range distribution (10 m bins)."""
import glob, os, sys
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\pravin\Desktop\PythonProject1\LiDAR data"
data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
if not files:
    sys.exit(f"No .csv files under {data_dir}")

ranges, n_pts, h_fov, v_fov, periods = [], [], [], [], []
n_total = n_nan = 0
last_folder, last_ts = None, None

for f in files:
    df = pd.read_csv(f, sep=";")
    n_total += df.size
    n_nan += int(df.isna().sum().sum())
    ranges.append(df["DISTANCE"].to_numpy(float))
    n_pts.append(len(df))
    az = np.degrees(np.arctan2(df["Y"], df["X"]))
    el = np.degrees(np.arctan2(df["Z"], np.hypot(df["X"], df["Y"])))
    h_fov.append(az.max() - az.min())
    v_fov.append(el.max() - el.min())
    # frame period = timestamp difference between consecutive frames of the same folder
    folder, ts = os.path.dirname(f), float(df["TIMESTAMP"].iloc[0])
    if folder == last_folder:
        periods.append((ts - last_ts) / 1e9)
    last_folder, last_ts = folder, ts

r = np.concatenate(ranges)
r = r[np.isfinite(r) & (r > 0)]
periods = np.array(periods)


def row(name, spec, v):
    v = np.asarray(v, float)
    return {"Metric": name, "Spec": spec, "N": len(v), "Mean": v.mean(),
            "StdDev": v.std(ddof=1) if len(v) > 1 else 0.0,
            "Min": v.min(), "Max": v.max(), "Median": np.median(v)}


table = pd.DataFrame([
    row("Points per frame", "-", n_pts),
    row("Range (m)", "5 - 250", r),
    row("Horizontal FOV", "<= 70", h_fov),
    row("Vertical FOV", "<= 30", v_fov),
    row("Update rate (Hz)", "1 - 30", 1 / periods),
    row("Frame period (s)", "-", periods),
])

print(f"Folders analyzed      {len({os.path.dirname(f) for f in files})}")
print(f"Total frames          {len(files)}")
print(f"Total points          {sum(n_pts)}")
print(f"Missing / NaN values  {n_nan}\n")
print(table.round(3).to_string(index=False))

hist = np.histogram(r, bins=np.arange(0, 270, 10))[0]
print("\nRange distribution (10 m bins)")
for i, n in enumerate(hist):
    if n:
        print(f"  {i*10:3d}-{i*10+10:3d} m {n:9,d} {'#' * int(40 * n / max(hist.max(), 1))}")