# recovery_patients_with_caregiver_baseline.py
# 只畫「病患」的恢復軌跡，並以「照護者」作為灰色 baseline 參考線
# 初階友善：pandas + matplotlib，線性流程、清楚註解

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 1) 讀檔 ----------------
physio = pd.read_csv("physiological_dataset_day.csv")   # steps/hr/sleep/activity + role
psych  = pd.read_csv("psych_behavioral_dataset_day.csv")# MOOD + role

# 確保 DaysFromTransplant 是數字
for df in [physio, psych]:
    if "DaysFromTransplant" in df.columns:
        df["DaysFromTransplant"] = pd.to_numeric(df["DaysFromTransplant"], errors="coerce")

# ---------------- 2) 基礎設定 ----------------
# 只分析存在的欄位（沒有就自動跳過）
metrics_physio = ["total_steps", "percent_active", "mean_hr", "sleep_duration"]
metric_psych   = "MOOD"

# 日樣本數門檻（避免極小樣本扭曲）
MIN_N_PER_DAY = 5

# 移動平均視窗（讓曲線順一點）
MA_WINDOW = 7   # 7-day moving average
MA_MINPTS = 3   # 至少 3 點才算

# ---------------- 3) 輔助函式 ----------------
def daily_mean_with_n(df, value_col):
    """回傳按 day 計算的 mean 與當天人數 n。"""
    agg = df.groupby("DaysFromTransplant")[value_col].agg(["mean","count"]).reset_index()
    agg = agg.rename(columns={"mean": value_col, "count": "n"})
    return agg

def add_ma(series, window=7, minpts=3):
    """加 7 天移動平均（中心對齊），不足點數時不硬湊。"""
    return series.rolling(window=window, center=True, min_periods=minpts).mean()

def plot_patients_with_caregiver_baseline(p_series, c_series, value_col, title, fname):
    """畫病患曲線 + 照護者 baseline（灰色），病患用主要顏色。"""
    plt.figure(figsize=(8,4.5))

    # 病患
    plt.plot(p_series["DaysFromTransplant"], p_series[value_col+"_ma"],
             label="Patients (7d MA)")

    # 照護者 baseline（灰色參考線）
    # 這裡使用「照護者的每日 7d MA 曲線」當作動態 baseline，
    # 若想用「照護者整體平均」改成一條水平線也可以（見下方備註）。
    if not c_series.empty:
        plt.plot(c_series["DaysFromTransplant"], c_series[value_col+"_ma"],
                 linestyle="--", alpha=0.6, color="gray", label="Caregivers baseline (7d MA)")

        # 若要水平線基準，改用這兩行：
        # cg_mean = c_series[value_col].mean()
        # plt.axhline(cg_mean, color="gray", linestyle="--", alpha=0.6, label=f"Caregivers mean ≈ {cg_mean:.1f}")

    plt.title(title)
    plt.xlabel("Days From Transplant")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ---------------- 4) 生理指標：病患曲線 + 照護者 baseline ----------------
if "role" in physio.columns and "DaysFromTransplant" in physio.columns:
    # 切病患/照護者
    pat = physio[physio["role"] == "Patients"].copy()
    cg  = physio[physio["role"] == "Caregivers"].copy()

    for m in metrics_physio:
        if m not in physio.columns:
            continue

        # 病患：按日平均 + 樣本數過濾 + 7日移動平均
        pat_daily = daily_mean_with_n(pat, m)
        pat_daily = pat_daily[pat_daily["n"] >= MIN_N_PER_DAY]
        if pat_daily.empty:
            continue
        pat_daily[m+"_ma"] = add_ma(pat_daily[m], window=MA_WINDOW, minpts=MA_MINPTS)

        # 照護者 baseline：同樣作法（但只作參考，不過濾也可以）
        cg_daily = daily_mean_with_n(cg, m)
        if not cg_daily.empty:
            cg_daily[m+"_ma"] = add_ma(cg_daily[m], window=MA_WINDOW, minpts=MA_MINPTS)

        # 畫圖
        plot_patients_with_caregiver_baseline(
            p_series=pat_daily,
            c_series=cg_daily,
            value_col=m,
            title=f"Patients Recovery vs Caregiver Baseline — {m}",
            fname=f"patients_recovery_{m}.png"
        )

# ---------------- 5) 心理指標（MOOD）：病患曲線 + 照護者 baseline ----------------
if metric_psych in psych.columns and "role" in psych.columns and "DaysFromTransplant" in psych.columns:
    pat_m = psych[psych["role"] == "Patients"].copy()
    cg_m  = psych[psych["role"] == "Caregivers"].copy()

    # 病患
    pat_daily = daily_mean_with_n(pat_m, metric_psych)
    pat_daily = pat_daily[pat_daily["n"] >= MIN_N_PER_DAY]
    if not pat_daily.empty:
        pat_daily[metric_psych+"_ma"] = add_ma(pat_daily[metric_psych], window=MA_WINDOW, minpts=MA_MINPTS)

        # 照護者 baseline
        cg_daily = daily_mean_with_n(cg_m, metric_psych)
        if not cg_daily.empty:
            cg_daily[metric_psych+"_ma"] = add_ma(cg_daily[metric_psych], window=MA_WINDOW, minpts=MA_MINPTS)

        plot_patients_with_caregiver_baseline(
            p_series=pat_daily,
            c_series=cg_daily,
            value_col=metric_psych,
            title="Patients Recovery vs Caregiver Baseline — MOOD",
            fname="patients_recovery_MOOD.png"
        )

print("Saved: patients_recovery_*.png")
