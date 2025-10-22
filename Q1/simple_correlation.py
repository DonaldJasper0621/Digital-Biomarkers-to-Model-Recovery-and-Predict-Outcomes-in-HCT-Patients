#simple_correlation.py
import pandas as pd

# 1) 讀資料
phys = pd.read_csv("../physiological_dataset_day.csv")
psych = pd.read_csv("../psych_behavioral_dataset_day.csv")

# 2) 衍生：睡眠效率
if {"ASLEEP_MIN", "INBED_VALUE"}.issubset(phys.columns):
    phys["sleep_efficiency"] = phys["ASLEEP_MIN"] / phys["INBED_VALUE"]

# 3) 同日合併（只看病患）
cols_phys = [
    "STUDY_PRTCPT_ID",
    "DaysFromTransplant",
    "total_steps",
    "percent_active",
    "sleep_efficiency",
    "mean_hr",
    "sleep_duration",
    "role",
]
cols_psych = [
    "STUDY_PRTCPT_ID",
    "DaysFromTransplant",
    "MOOD",
    "t_fatig",
    "t_deprss",
    "role",
]

phys = phys[[c for c in cols_phys if c in phys.columns]]
psych = psych[[c for c in cols_psych if c in psych.columns]]

merged = phys.merge(
    psych, on=["STUDY_PRTCPT_ID", "DaysFromTransplant"], suffixes=("_phys", "_psych")
)
merged = merged[merged["role_phys"] == "Patients"].copy()

# 4) 選要做相關的變數
phys_vars = [
    c
    for c in [
        "sleep_efficiency",
        "percent_active",
        "total_steps",
        "mean_hr",
        "sleep_duration",
    ]
    if c in merged.columns
]
psych_vars = [c for c in ["MOOD", "t_fatig", "t_deprss"] if c in merged.columns]

rows = []
for x in phys_vars:
    for y in psych_vars:
        sub = merged[[x, y]].dropna()
        n = len(sub)
        if n >= 20:
            # pandas 內建皮爾森相關
            r = sub[x].corr(sub[y], method="pearson")
            rows.append({"x_phys": x, "y_psych": y, "n": n, "r": r})

out = pd.DataFrame(rows)
out.to_csv("q2_overall_correlations.csv", index=False)
print(out.sort_values("r", ascending=False).head(10))
print("Saved: q2_overall_correlations.csv")
