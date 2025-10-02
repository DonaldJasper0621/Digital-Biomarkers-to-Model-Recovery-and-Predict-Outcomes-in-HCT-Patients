# === Milestone 1a: Intermediate Starter Pipeline ===
# Builds two joinable datasets and saves clean CSVs.

import pandas as pd
import numpy as np
from pathlib import Path

# 0) Paths ----------------------------------------------------------
DATA_DIR = Path(".")  # change to your folder if needed


def read_csv(name):
    path = DATA_DIR / name
    return pd.read_csv(path)


# 1) Load raw tables ------------------------------------------------
demographic = read_csv("./Initial_data/demographic_data.csv")
daily_steps = read_csv("./Initial_data/daily_steps.csv")
daily_activity = read_csv("./Initial_data/daily_activity.csv")
sleep_classic = read_csv("./Initial_data/sleep_classic.csv")
sleep_stages = read_csv("./Initial_data/sleep_stages.csv")
mood = read_csv("./Initial_data/mood.csv")
promis = read_csv("./Initial_data/PROMIS_tscore.csv")
outcome = read_csv("./Initial_data/outcome.csv")
infections = read_csv("./Initial_data/infections.csv")

# Optional (if present in your drop)
try:
    daily_hr = read_csv("./Initial_data/daily_hr.csv")
except FileNotFoundError:
    daily_hr = pd.DataFrame()

# 2) Light cleaning -------------------------------------------------
tables = [
    demographic,
    daily_steps,
    daily_activity,
    sleep_classic,
    sleep_stages,
    mood,
    promis,
    outcome,
    infections,
    daily_hr,
]

for df in tables:
    if not df.empty and "STUDY_PRTCPT_ID" in df.columns:
        df["STUDY_PRTCPT_ID"] = df["STUDY_PRTCPT_ID"].astype(str).str.strip()

for df in [daily_steps, daily_activity, sleep_classic, sleep_stages, mood, daily_hr]:
    if not df.empty and "DaysFromTransplant" in df.columns:
        df["DaysFromTransplant"] = pd.to_numeric(
            df["DaysFromTransplant"], errors="coerce"
        )

# PROMIS labeled timepoints → approximate day index
if "Timestamp" in promis.columns:
    promis["DaysFromTransplant_PROMIS"] = promis["Timestamp"].map(
        {"Baseline": 0, "Day30": 30, "Day120": 120}
    )

# 3) Dataset 1: Physiological (day-level) ---------------------------
# Backbone = daily_steps; change here if you prefer daily_activity as base.
keys = ["STUDY_PRTCPT_ID", "DaysFromTransplant"]
physio = daily_steps.copy()


def take_cols(df, keep):
    cols = [c for c in keep if c in df.columns]
    return df[cols]


# Merge: daily_activity
act_cols = keys + [
    "sedentary",
    "lightly_active",
    "moderately_active",
    "very_active",
    "total_active_time",
    "total_measured_time",
    "percent_sedentary",
    "percent_active",
    "time_coverage",
]
if not daily_activity.empty:
    physio = physio.merge(take_cols(daily_activity, act_cols), on=keys, how="outer")

# Merge: sleep_stages
stg_cols = keys + [
    "DEEP_MIN",
    "LIGHT_MIN",
    "REM_MIN",
    "WAKE_MIN",
    "DEEP_COUNT",
    "LIGHT_COUNT",
    "REM_COUNT",
    "WAKE_COUNT",
]
if not sleep_stages.empty:
    physio = physio.merge(take_cols(sleep_stages, stg_cols), on=keys, how="outer")

# Merge: sleep_classic
cls_cols = keys + [
    "sleep_duration",
    "ASLEEP_VALUE",
    "INBED_VALUE",
    "ASLEEP_MIN",
    "ASLEEP_COUNT",
    "AWAKE_COUNT",
    "AWAKE_MIN",
    "RESTLESS_COUNT",
    "RESTLESS_MIN",
]
if not sleep_classic.empty:
    physio = physio.merge(take_cols(sleep_classic, cls_cols), on=keys, how="outer")

# Merge: daily_hr (optional)
hr_cols = keys + [
    "mean_hr",
    "median_hr",
    "min_hr",
    "max_hr",
    "sd_hr",
    "morning_hr",
    "afternoon_hr",
    "evening_hr",
    "night_hr",
    "time_coverage",
]
if not daily_hr.empty:
    physio = physio.merge(take_cols(daily_hr, hr_cols), on=keys, how="left")

# Add demographics
demo_keep = [
    "STUDY_PRTCPT_ID",
    "age",
    "gender",
    "role",
    "arm",
    "monthly_income",
    "cg_hours",
    "transplant_type",
]
for col in demo_keep:
    if col not in demographic.columns:
        demographic[col] = np.nan

physio = (
    physio.merge(
        demographic[demo_keep].drop_duplicates("STUDY_PRTCPT_ID"),
        on="STUDY_PRTCPT_ID",
        how="left",
    )
    .sort_values(keys)
    .reset_index(drop=True)
)

# 4) Dataset 2: Psychological/Behavioral (day-level) ----------------
psych = mood.copy()
psych = psych.merge(
    demographic[demo_keep].drop_duplicates("STUDY_PRTCPT_ID"),
    on="STUDY_PRTCPT_ID",
    how="left",
)

# PROMIS joins on ID + Group; remains at Baseline/Day30/Day120
if set(["STUDY_PRTCPT_ID", "Group"]).issubset(promis.columns):
    p_cols = ["STUDY_PRTCPT_ID", "Group", "DaysFromTransplant_PROMIS"] + [
        c for c in promis.columns if c.startswith("t_")
    ]
    psych = psych.merge(promis[p_cols], on=["STUDY_PRTCPT_ID", "Group"], how="left")
    if "DaysFromTransplant" in psych.columns:
        psych["is_promis_day"] = (
            psych["DaysFromTransplant"] == psych["DaysFromTransplant_PROMIS"]
        ).astype(int)

psych = psych.sort_values(keys).reset_index(drop=True)

# 5) Events standardization ----------------------------------------
# infections: rename date → DaysFromTransplant
if "date_culture_drawn" in infections.columns:
    infections_std = infections.rename(
        columns={"date_culture_drawn": "DaysFromTransplant"}
    )
else:
    infections_std = infections.copy()

if "DaysFromTransplant" in infections_std.columns:
    infections_std["DaysFromTransplant"] = pd.to_numeric(
        infections_std["DaysFromTransplant"], errors="coerce"
    )

# outcomes: ensure *_date numeric
outcomes_std = outcome.copy()
for c in outcomes_std.columns:
    if c.endswith("_date"):
        outcomes_std[c] = pd.to_numeric(outcomes_std[c], errors="coerce")

# 6) Save outputs ---------------------------------------------------
physio_out = DATA_DIR / "physiological_dataset_day.csv"
psych_out = DATA_DIR / "psych_behavioral_dataset_day.csv"
inf_out = DATA_DIR / "events_infections.csv"
out_out = DATA_DIR / "events_outcomes.csv"

physio.to_csv(physio_out, index=False)
psych.to_csv(psych_out, index=False)
infections_std.to_csv(inf_out, index=False)
outcomes_std.to_csv(out_out, index=False)

print("Saved:")
print(physio_out)
print(psych_out)
print(inf_out)
print(out_out)

