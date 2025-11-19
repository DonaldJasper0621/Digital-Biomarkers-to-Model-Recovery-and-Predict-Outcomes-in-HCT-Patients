# recovery_patients_with_caregiver_baseline.py
# Plot patients’ recovery trajectories with caregivers as a gray baseline.
# Beginner-friendly version: clean, linear flow using pandas + matplotlib.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#  1) Read input files 
physio = pd.read_csv("physiological_dataset_day.csv")   # steps/hr/sleep/activity + role
psych  = pd.read_csv("psych_behavioral_dataset_day.csv")# mood + role

# Output directory for all figures
OUT_DIR = Path("Trajectory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure "DaysFromTransplant" is numeric
for df in [physio, psych]:
    if "DaysFromTransplant" in df.columns:
        df["DaysFromTransplant"] = pd.to_numeric(df["DaysFromTransplant"], errors="coerce")

#  2) Basic configuration 
# Only analyze columns that actually exist in the dataset
metrics_physio = ["total_steps", "percent_active", "mean_hr", "sleep_duration"]
metric_psych   = "MOOD"

# Minimum daily sample size (to avoid distortion from tiny samples)
MIN_N_PER_DAY = 5

# Moving-average settings (to smooth the curves)
MA_WINDOW = 7   # 7-day moving average
MA_MINPTS = 3   # Require at least 3 data points to calculate MA

# 3) Helper functions 
def daily_mean_with_n(df, value_col):
    """
    Compute daily mean and sample count (n) for a given column.
    Returns a DataFrame with columns: DaysFromTransplant, <value_col>, n
    """
    agg = df.groupby("DaysFromTransplant")[value_col].agg(["mean", "count"]).reset_index()
    agg = agg.rename(columns={"mean": value_col, "count": "n"})
    return agg


def add_ma(series, window=7, minpts=3):
    """
    Apply a centered moving average to smooth data.
    Only compute when at least `minpts` points are available.
    """
    return series.rolling(window=window, center=True, min_periods=minpts).mean()


def plot_patients_with_caregiver_baseline(p_series, c_series, value_col, title, fname):
    """
    Plot patients’ recovery curve (colored) and caregivers’ baseline (gray dashed line).
    """
    plt.figure(figsize=(8, 4.5))

    # Patients’ curve
    plt.plot(
        p_series["DaysFromTransplant"],
        p_series[value_col + "_ma"],
        label="Patients (7-day MA)"
    )

    # Caregiver baseline (gray dashed reference line)
    # Uses caregivers’ daily 7-day MA as a dynamic baseline.
    if not c_series.empty:
        plt.plot(
            c_series["DaysFromTransplant"],
            c_series[value_col + "_ma"],
            linestyle="--",
            alpha=0.6,
            color="gray",
            label="Caregivers baseline (7-day MA)"
        )

    plt.title(title)
    plt.xlabel("Days From Transplant")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fname), dpi=150)
    plt.close()

#  4) Physiological metrics: patients vs caregivers 
if "role" in physio.columns and "DaysFromTransplant" in physio.columns:
    # Split data into patients and caregivers
    pat = physio[physio["role"] == "Patients"].copy()
    cg  = physio[physio["role"] == "Caregivers"].copy()

    for m in metrics_physio:
        if m not in physio.columns:
            continue

        # Patients: compute daily mean, filter by sample size, apply moving average
        pat_daily = daily_mean_with_n(pat, m)
        pat_daily = pat_daily[pat_daily["n"] >= MIN_N_PER_DAY]
        if pat_daily.empty:
            continue
        pat_daily[m + "_ma"] = add_ma(pat_daily[m], window=MA_WINDOW, minpts=MA_MINPTS)

        # Caregivers baseline: same procedure (optional filtering)
        cg_daily = daily_mean_with_n(cg, m)
        if not cg_daily.empty:
            cg_daily[m + "_ma"] = add_ma(cg_daily[m], window=MA_WINDOW, minpts=MA_MINPTS)

        # Plot both
        plot_patients_with_caregiver_baseline(
            p_series=pat_daily,
            c_series=cg_daily,
            value_col=m,
            title=f"Patients Recovery vs Caregiver Baseline — {m}",
            fname=OUT_DIR / f"patients_recovery_{m}.png"
        )

# 5) Psychological metric (MOOD): patients vs caregivers 
if metric_psych in psych.columns and "role" in psych.columns and "DaysFromTransplant" in psych.columns:
    pat_m = psych[psych["role"] == "Patients"].copy()
    cg_m  = psych[psych["role"] == "Caregivers"].copy()

    # Patients
    pat_daily = daily_mean_with_n(pat_m, metric_psych)
    pat_daily = pat_daily[pat_daily["n"] >= MIN_N_PER_DAY]
    if not pat_daily.empty:
        pat_daily[metric_psych + "_ma"] = add_ma(pat_daily[metric_psych], window=MA_WINDOW, minpts=MA_MINPTS)

        # Caregivers
        cg_daily = daily_mean_with_n(cg_m, metric_psych)
        if not cg_daily.empty:
            cg_daily[metric_psych + "_ma"] = add_ma(cg_daily[metric_psych], window=MA_WINDOW, minpts=MA_MINPTS)

        plot_patients_with_caregiver_baseline(
            p_series=pat_daily,
            c_series=cg_daily,
            value_col=metric_psych,
            title="Patients Recovery vs Caregiver Baseline — MOOD",
            fname=OUT_DIR / "patients_recovery_MOOD.png"
        )

print("Saved: patients_recovery_*.png")
print(f"Saved plots to: {OUT_DIR.resolve()}")
