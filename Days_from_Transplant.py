# recovery_trajectories_basic.py
# 初階版：畫出不同群體(病患/照護者)隨「DaysFromTransplant」的平均變化曲線

import pandas as pd
import matplotlib.pyplot as plt

# 1) 讀資料 ---------------------------------------------------------
physio = pd.read_csv("physiological_dataset_day.csv")   # 生理: steps/hr/sleep/activity + role
psych  = pd.read_csv("psych_behavioral_dataset_day.csv")# 心理: MOOD + role

# 2) 選指標(存在才會畫) ---------------------------------------------
metrics_physio = ["total_steps", "percent_active", "mean_hr", "sleep_duration"]
metric_psych   = "MOOD"

# 3) 確保欄位存在與資料型別 -----------------------------------------
for df in [physio, psych]:
    if "DaysFromTransplant" in df.columns:
        df["DaysFromTransplant"] = pd.to_numeric(df["DaysFromTransplant"], errors="coerce")

# 4) 畫 生理指標 的恢復曲線 ------------------------------------------
if "role" in physio.columns and "DaysFromTransplant" in physio.columns:
    for m in metrics_physio:
        if m in physio.columns:
            # groupby day & role, 取平均
            gp = physio.groupby(["DaysFromTransplant", "role"], as_index=False)[m].mean().dropna()

            # 簡單折線圖：Patients vs Caregivers
            plt.figure(figsize=(7,4))
            for group_name in gp["role"].dropna().unique():
                sub = gp[gp["role"] == group_name]
                plt.plot(sub["DaysFromTransplant"], sub[m], label=group_name)

            plt.title(f"Recovery Trajectory — {m} (mean by day)")
            plt.xlabel("Days From Transplant")
            plt.ylabel(m)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"traj_{m}_by_role.png", dpi=150)
            plt.close()

# 5) 畫 心理指標(MOOD) 的恢復曲線 ------------------------------------
if metric_psych in psych.columns and "DaysFromTransplant" in psych.columns and "role" in psych.columns:
    gp_mood = psych.groupby(["DaysFromTransplant","role"], as_index=False)[metric_psych].mean().dropna()

    plt.figure(figsize=(7,4))
    for group_name in gp_mood["role"].dropna().unique():
        sub = gp_mood[gp_mood["role"] == group_name]
        plt.plot(sub["DaysFromTransplant"], sub[metric_psych], label=group_name)

    plt.title("Recovery Trajectory — MOOD (mean by day)")
    plt.xlabel("Days From Transplant")
    plt.ylabel("MOOD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("traj_MOOD_by_role.png", dpi=150)
    plt.close()

print("Saved: traj_*.png")
