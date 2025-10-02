# group_comparisons.py
# Compare group differences (Patients vs Caregivers, Male vs Female)
# Intermediate-level script with Welch t-test, Cohen's d, FDR correction, and plots.

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# ------------ Config ------------
DATA_DIR = Path(".")
PHYSIO_FILE = DATA_DIR / "physiological_dataset_day.csv"
PSYCH_FILE  = DATA_DIR / "psych_behavioral_dataset_day.csv"
OUT_DIR     = DATA_DIR / "eda_group_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Choose metrics to compare (存在才會分析，不存在會自動跳過)
PHYSIO_METRICS = [
    "total_steps", "mean_hr", "sleep_duration",  # 基本欄位
    # 延伸欄位（可能存在）
    "percent_active", "ASLEEP_MIN", "INBED_VALUE"
]
PSYCH_METRICS = [
    "MOOD"
]

# ------------ Helpers ------------
def cohen_d(x, y):
    """Cohen's d with pooled SD. Returns np.nan if insufficient data."""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    if sp2 <= 0 or np.isnan(sp2):
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    return d

def welch_ttest(x, y):
    """Welch t-test (unequal variances). Returns t, p."""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(t), float(p)

def bh_fdr(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR correction. Returns (reject, p_adjusted)."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.arange(1, n+1)
    p_adj = np.empty(n, dtype=float)
    p_adj[order] = np.minimum.accumulate((pvals[order] * n) / ranked[::-1])[::-1]
    p_adj = np.clip(p_adj, 0, 1)
    reject = p_adj <= alpha
    return reject, p_adj

def add_sleep_efficiency(df):
    """Create sleep_efficiency = ASLEEP_MIN / INBED_VALUE if both exist."""
    if "ASLEEP_MIN" in df.columns and "INBED_VALUE" in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["sleep_efficiency"] = df["ASLEEP_MIN"] / df["INBED_VALUE"]
    return df

def clean_demo_cols(df):
    # 標準化關鍵欄位，避免大小寫/空白問題
    for c in ["role","gender","transplant_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def quick_boxplot(df, value_col, group_col, title, fname):
    plt.figure(figsize=(6,4))
    # 使用 pandas 簡單 boxplot，避免額外依賴
    df[[value_col, group_col]].dropna().boxplot(by=group_col, column=value_col)
    plt.title(title)
    plt.suptitle("")  # 去掉上方副標
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()

def quick_bar_means(df, value_col, group_col, title, fname):
    g = df[[value_col, group_col]].dropna().groupby(group_col)[value_col].mean()
    plt.figure(figsize=(6,4))
    g.plot(kind="bar")
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(f"Mean {value_col}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()

def compare_groups(df, metric, group_col, g1, g2):
    """Return summary dict for metric comparing g1 vs g2 on group_col."""
    if metric not in df.columns or group_col not in df.columns:
        return None
    a = df.loc[df[group_col] == g1, metric].dropna()
    b = df.loc[df[group_col] == g2, metric].dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = welch_ttest(a, b)
    d = cohen_d(a, b)
    return {
        "metric": metric,
        "group_col": group_col,
        "group1": g1,
        "group2": g2,
        "n1": len(a),
        "n2": len(b),
        "mean1": a.mean(),
        "sd1": a.std(ddof=1),
        "mean2": b.mean(),
        "sd2": b.std(ddof=1),
        "t_stat": t,
        "p_value": p,
        "cohens_d": d
    }

# ------------ Load & prep ------------
physio = pd.read_csv(PHYSIO_FILE)
psych  = pd.read_csv(PSYCH_FILE)

# 清理與衍生欄位
physio = clean_demo_cols(add_sleep_efficiency(physio))
psych  = clean_demo_cols(psych)

# 針對要分析的欄位做「存在性」過濾
physio_metrics_exist = [m for m in PHYSIO_METRICS + ["sleep_efficiency"] if m in physio.columns]
psych_metrics_exist  = [m for m in PSYCH_METRICS if m in psych.columns]

# ------------ Patients vs Caregivers ------------
results_role = []

# Physiological 指標
for m in physio_metrics_exist:
    R = compare_groups(physio, m, "role", "Patients", "Caregivers")
    if R: results_role.append(R)

# Psychological 指標（MOOD）
for m in psych_metrics_exist:
    R = compare_groups(psych, m, "role", "Patients", "Caregivers")
    if R: results_role.append(R)

# FDR 校正（針對本區塊所有檢定）
if results_role:
    pvals = [r["p_value"] for r in results_role]
    reject, p_adj = bh_fdr(pvals, alpha=0.05)
    for i, r in enumerate(results_role):
        r["p_adj_fdr"] = p_adj[i]
        r["reject_fdr"] = bool(reject[i])

df_role = pd.DataFrame(results_role)
df_role.to_csv(OUT_DIR / "compare_role_patients_vs_caregivers.csv", index=False)

# 畫圖（平均值長條 + 箱型圖）
for m in physio_metrics_exist:
    quick_bar_means(physio, m, "role", f"Mean {m}: Patients vs Caregivers", f"bar_{m}_role.png")
    quick_boxplot(physio, m, "role", f"{m}: Patients vs Caregivers", f"box_{m}_role.png")
for m in psych_metrics_exist:
    quick_bar_means(psych, m, "role", f"Mean {m}: Patients vs Caregivers", f"bar_{m}_role.png")
    quick_boxplot(psych, m, "role", f"{m}: Patients vs Caregivers", f"box_{m}_role.png")

# ------------ Male vs Female ------------
results_gender = []

# 生理
if "gender" in physio.columns:
    for m in physio_metrics_exist:
        R = compare_groups(physio, m, "gender", "Male", "Female")
        if R: results_gender.append(R)

# 心理
if "gender" in psych.columns:
    for m in psych_metrics_exist:
        R = compare_groups(psych, m, "gender", "Male", "Female")
        if R: results_gender.append(R)

if results_gender:
    pvals = [r["p_value"] for r in results_gender]
    reject, p_adj = bh_fdr(pvals, alpha=0.05)
    for i, r in enumerate(results_gender):
        r["p_adj_fdr"] = p_adj[i]
        r["reject_fdr"] = bool(reject[i])

df_gender = pd.DataFrame(results_gender)
df_gender.to_csv(OUT_DIR / "compare_gender_male_vs_female.csv", index=False)

# 圖
if "gender" in physio.columns:
    for m in physio_metrics_exist:
        quick_bar_means(physio, m, "gender", f"Mean {m}: Male vs Female", f"bar_{m}_gender.png")
        quick_boxplot(physio, m, "gender", f"{m}: Male vs Female", f"box_{m}_gender.png")
if "gender" in psych.columns:
    for m in psych_metrics_exist:
        quick_bar_means(psych, m, "gender", f"Mean {m}: Male vs Female", f"bar_{m}_gender.png")
        quick_boxplot(psych, m, "gender", f"{m}: Male vs Female", f"box_{m}_gender.png")
