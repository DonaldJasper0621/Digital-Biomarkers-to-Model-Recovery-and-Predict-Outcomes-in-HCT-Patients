# === Milestone 1a: Intermediate Starter Pipeline ===
# Builds two joinable datasets and saves clean CSVs.
# build_two_datasets_simple.py
# 目的：把原始檔合成兩個日級別資料集：physiological / psychological + 事件表
# 重點：使用 keys union 當骨架，避免漏資料；僅 pandas/numpy

import pandas as pd
import numpy as np

DATA_DIR = "./Initial_data"


# -------- 讀檔（不存在就回空表） ----------
def read_or_empty(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


demographic = read_or_empty(f"{DATA_DIR}/demographic_data.csv")
daily_steps = read_or_empty(f"{DATA_DIR}/daily_steps.csv")
daily_activity = read_or_empty(f"{DATA_DIR}/daily_activity.csv")
sleep_classic = read_or_empty(f"{DATA_DIR}/sleep_classic.csv")
sleep_stages = read_or_empty(f"{DATA_DIR}/sleep_stages.csv")
mood = read_or_empty(f"{DATA_DIR}/mood.csv")
promis = read_or_empty(f"{DATA_DIR}/PROMIS_tscore.csv")
outcome = read_or_empty(f"{DATA_DIR}/outcome.csv")
infections = read_or_empty(f"{DATA_DIR}/infections.csv")
daily_hr = read_or_empty(f"{DATA_DIR}/daily_hr.csv")

# -------- 輕清理 ----------
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
        df["STUDY_PRTCPT_ID"] = df["STUDY_PRTCPT_ID"].astype(str).strip()


def to_num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


for df in [daily_steps, daily_activity, sleep_classic, sleep_stages, mood, daily_hr]:
    to_num(df, "DaysFromTransplant")

if "Timestamp" in promis.columns:
    promis["DaysFromTransplant_PROMIS"] = promis["Timestamp"].map(
        {"Baseline": 0, "Day30": 30, "Day120": 120}
    )

# -------- 用 keys union 當骨架（避免漏掉沒有 steps 的日子） ----------
keys = ["STUDY_PRTCPT_ID", "DaysFromTransplant"]
bases = []
for d in [daily_steps, daily_activity, sleep_classic, sleep_stages, daily_hr]:
    if not d.empty and set(keys).issubset(d.columns):
        bases.append(d[keys])
base = (
    pd.concat(bases, axis=0).drop_duplicates() if bases else pd.DataFrame(columns=keys)
)

# -------- 建 Dataset 1: Physiological（日） ----------
physio = base.copy()


def take(df, cols):
    keep = [c for c in cols if c in df.columns]
    return df[keep] if keep else pd.DataFrame(columns=cols)


# steps / activity / sleep / hr 依序 left merge 到骨架
if not daily_steps.empty:
    physio = physio.merge(
        take(
            daily_steps,
            keys + ["total_steps", "n_measurements", "time_coverage", "Group"],
        ),
        on=keys,
        how="left",
    )

if not daily_activity.empty:
    physio = physio.merge(
        take(
            daily_activity,
            keys
            + [
                "percent_active",
                "sedentary",
                "lightly_active",
                "moderately_active",
                "very_active",
                "time_coverage",
                "Group",
            ],
        ),
        on=keys,
        how="left",
    )

if not sleep_classic.empty:
    physio = physio.merge(
        take(
            sleep_classic,
            keys + ["sleep_duration", "ASLEEP_MIN", "INBED_VALUE", "Group"],
        ),
        on=keys,
        how="left",
    )

if not sleep_stages.empty:
    physio = physio.merge(
        take(
            sleep_stages,
            keys
            + [
                "DEEP_MIN",
                "LIGHT_MIN",
                "REM_MIN",
                "WAKE_MIN",
                "DEEP_COUNT",
                "LIGHT_COUNT",
                "REM_COUNT",
                "WAKE_COUNT",
                "Group",
            ],
        ),
        on=keys,
        how="left",
    )

if not daily_hr.empty:
    physio = physio.merge(
        take(
            daily_hr,
            keys
            + [
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
                "Group",
            ],
        ),
        on=keys,
        how="left",
    )

# 衍生：睡眠效率
if {"ASLEEP_MIN", "INBED_VALUE"}.issubset(physio.columns):
    with np.errstate(divide="ignore", invalid="ignore"):
        physio["sleep_efficiency"] = physio["ASLEEP_MIN"] / physio["INBED_VALUE"]

# demographics（只保留一筆）
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
for c in demo_keep:
    if c not in demographic.columns:
        demographic[c] = np.nan

physio = (
    physio.merge(
        demographic[demo_keep].drop_duplicates("STUDY_PRTCPT_ID"),
        on="STUDY_PRTCPT_ID",
        how="left",
    )
    .sort_values(keys)
    .reset_index(drop=True)
)

# -------- 建 Dataset 2: Psychological/Behavioral（日） ----------
psych = mood.copy()
if not psych.empty:
    psych = psych.merge(
        demographic[demo_keep].drop_duplicates("STUDY_PRTCPT_ID"),
        on="STUDY_PRTCPT_ID",
        how="left",
    )
    # PROMIS（仍保留在個別 timepoint，給對照用）
    if set(["STUDY_PRTCPT_ID", "Group"]).issubset(promis.columns):
        tcols = ["STUDY_PRTCPT_ID", "Group", "DaysFromTransplant_PROMIS"] + [
            c for c in promis.columns if c.startswith("t_")
        ]
        tcols = [c for c in tcols if c in promis.columns]
        psych = psych.merge(promis[tcols], on=["STUDY_PRTCPT_ID", "Group"], how="left")
        if "DaysFromTransplant" in psych.columns:
            psych["is_promis_day"] = (
                psych["DaysFromTransplant"] == psych["DaysFromTransplant_PROMIS"]
            ).astype(int)

psych = psych.sort_values(keys).reset_index(drop=True)

# -------- 事件表標準化 ----------
# infections: date_culture_drawn -> DaysFromTransplant
if "date_culture_drawn" in infections.columns:
    infections_std = infections.rename(
        columns={"date_culture_drawn": "DaysFromTransplant"}
    )
else:
    infections_std = infections.copy()
to_num(infections_std, "DaysFromTransplant")

# outcomes: *_date 轉數字
outcomes_std = outcome.copy()
for c in outcomes_std.columns:
    if c.endswith("_date"):
        to_num(outcomes_std, c)

# -------- 輸出 ----------
physio.to_csv("physiological_dataset_day.csv", index=False)
psych.to_csv("psych_behavioral_dataset_day.csv", index=False)
infections_std.to_csv("events_infections.csv", index=False)
outcomes_std.to_csv("events_outcomes.csv", index=False)

print("Saved:")
print("  physiological_dataset_day.csv")
print("  psych_behavioral_dataset_day.csv")
print("  events_infections.csv")
print("  events_outcomes.csv")
