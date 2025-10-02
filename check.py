# check_heads_and_rationale.py
# Minimal script to look at outputs and write rationale.

import pandas as pd
import numpy as np

# --- 0) Load the outputs you just built ---
physio = pd.read_csv("physiological_dataset_day.csv")
psych = pd.read_csv("psych_behavioral_dataset_day.csv")
infx = pd.read_csv("events_infections.csv")
outc = pd.read_csv("events_outcomes.csv")

# --- 1) Print heads and shapes ---
print("\n[physio head]")
print(physio.head())
print("physio shape:", physio.shape)

print("\n[psych head]")
print(psych.head())
print("psych shape:", psych.shape)

print("\n[infections head]")
print(infx.head())
print("infections shape:", infx.shape)

print("\n[outcomes head]")
print(outc.head())
print("outcomes shape:", outc.shape)

# --- 2) Same-day merge (align physio with MOOD) ---
keys = ["STUDY_PRTCPT_ID", "DaysFromTransplant"]
keep_psych = [
    c
    for c in ["STUDY_PRTCPT_ID", "DaysFromTransplant", "MOOD", "Group"]
    if c in psych.columns
]
same_day = physio.merge(psych[keep_psych], on=keys, how="inner")

print("\n[same_day merged head]")
print(same_day.head())
print("same_day shape:", same_day.shape)

# --- 3) Simple comparisons (only if columns exist) ---
# corr(total_steps, MOOD)
if "total_steps" in same_day.columns and "MOOD" in same_day.columns:
    corr_steps_mood = same_day[["total_steps", "MOOD"]].corr().iloc[0, 1]
else:
    corr_steps_mood = None

# MOOD by activity tertiles
if "percent_active" in same_day.columns and "MOOD" in same_day.columns:
    tertile = pd.qcut(
        same_day["percent_active"].rank(method="first"),
        3,
        labels=["low", "mid", "high"],
    )
    mood_by_tert = (
        same_day.assign(activity_tertile=tertile)
        .groupby("activity_tertile")["MOOD"]
        .mean()
        .round(2)
    )
else:
    mood_by_tert = pd.Series(dtype=float)

# corr(sleep_efficiency, MOOD)
if set(["ASLEEP_MIN", "INBED_VALUE", "MOOD"]).issubset(same_day.columns):
    with np.errstate(divide="ignore", invalid="ignore"):
        sleep_eff = same_day["ASLEEP_MIN"] / same_day["INBED_VALUE"]
    corr_sleep_eff_mood = (
        pd.concat([sleep_eff.rename("sleep_efficiency"), same_day["MOOD"]], axis=1)
        .corr()
        .iloc[0, 1]
    )
else:
    corr_sleep_eff_mood = None

print("\n[quick stats]")
print("corr(total_steps, MOOD):", corr_steps_mood)
print("corr(sleep_efficiency, MOOD):", corr_sleep_eff_mood)
if not mood_by_tert.empty:
    print("avg MOOD by activity tertile:\n", mood_by_tert)

# --- 4) Write your Rationale to a text file ---
RATIONALE = """Rationale:
We selected these two datasets because they capture complementary dimensions of post-transplant recovery: physical functioning and psychological health. The physiological dataset provides continuous, objective measures from wearable devices, while the psychological/behavioral dataset offers self-reported outcomes and clinical events. Analyzing them together allows us to investigate whether changes in daily activity, heart rate, or sleep patterns are associated with mood, fatigue, or medical complications such as readmission and infection.
This split also simplifies the data integration process while maintaining analytical depth. It creates a clear framework for exploring clinically relevant questions, such as whether early deviations in physiological signals can predict later declines in psychological well-being or adverse clinical outcomes. By joining these datasets, we can provide insights that support improved monitoring and intervention strategies for both patients and caregivers following transplantation.
"""

with open("Rationale.txt", "w") as f:
    f.write(RATIONALE)

print("\nWrote Rationale.txt")
