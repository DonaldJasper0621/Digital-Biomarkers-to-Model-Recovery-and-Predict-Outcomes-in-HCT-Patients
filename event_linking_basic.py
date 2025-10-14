# early_warning_signs.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
DATA_DIR   = Path(".")
PHYSIO_CSV = DATA_DIR / "physiological_dataset_day.csv"
EVENTS_CSV = DATA_DIR / "events_infections.csv"
OUT_DIR    = DATA_DIR / "early_warning_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ("percent_active", "total_steps", "mean_hr", "sleep_duration")

# windows: relative to infection event day
PRE_START, PRE_END   = -7,  -1
BASE_START, BASE_END = -30, -14

# flag thresholds
PCT_DROP_THRESHOLD = -0.15
SD_DROP_THRESHOLD  = -0.50

PLOT = True  # 關掉圖就設 False

# ---------------- Core ----------------
@dataclass(frozen=True)
class Windows:
    pre_start: int = PRE_START
    pre_end:   int = PRE_END
    base_start:int = BASE_START
    base_end:  int = BASE_END

def _clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "STUDY_PRTCPT_ID" in df.columns:
        df = df.assign(STUDY_PRTCPT_ID=df["STUDY_PRTCPT_ID"].astype(str).str.strip())
    if "DaysFromTransplant" in df.columns:
        df = df.assign(DaysFromTransplant=pd.to_numeric(df["DaysFromTransplant"], errors="coerce"))
    return df

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    physio = _clean_ids(pd.read_csv(PHYSIO_CSV))
    events = _clean_ids(pd.read_csv(EVENTS_CSV))
    events = events.rename(columns={"DaysFromTransplant": "event_day"})
    events = events[["STUDY_PRTCPT_ID", "event_day"]].dropna()
    return physio, events

def summarize_event(
    p: pd.DataFrame, event_day: int, metrics: tuple[str, ...], w: Windows
) -> list[dict]:
    rows = []
    day = p["DaysFromTransplant"]

    pre  = p[(day >= event_day + w.pre_start)  & (day <= event_day + w.pre_end)]
    base = p[(day >= event_day + w.base_start) & (day <= event_day + w.base_end)]
    if len(pre) < 2 or len(base) < 2:
        return rows

    for m in metrics:
        if m not in p.columns:
            continue

        pre_vals  = pre[m].dropna()
        base_vals = base[m].dropna()
        if len(pre_vals) < 2 or len(base_vals) < 2:
            continue

        b_mean = base_vals.mean()
        b_sd   = base_vals.std(ddof=1)
        p_mean = pre_vals.mean()

        # avoid divide-by-zero / NaN
        pct   = (p_mean - b_mean) / b_mean if (pd.notna(b_mean) and b_mean != 0) else np.nan
        sdchg = (p_mean - b_mean) / b_sd   if (pd.notna(b_sd) and b_sd  > 0) else np.nan

        rows.append({
            "metric": m,
            "baseline_mean": b_mean,
            "baseline_sd":   b_sd,
            "pre7_mean":     p_mean,
            "delta_pre_minus_base": p_mean - b_mean,
            "pct_change": pct,
            "sd_change":  sdchg,
            "warn_pct_drop": bool(pd.notna(pct)  and pct  <= PCT_DROP_THRESHOLD),
            "warn_sd_drop":  bool(pd.notna(sdchg) and sdchg <= SD_DROP_THRESHOLD),
        })
    return rows

def compute_summary(physio: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, ev in events.iterrows():
        pid, e_day = ev["STUDY_PRTCPT_ID"], int(ev["event_day"])
        p = physio.loc[physio["STUDY_PRTCPT_ID"] == pid]
        if p.empty:
            continue
        rows = summarize_event(p, e_day, METRICS, Windows())
        for r in rows:
            r.update({"STUDY_PRTCPT_ID": pid, "event_day": e_day})
            out.append(r)
    return pd.DataFrame(out)

def build_overview(summary: pd.DataFrame) -> pd.DataFrame:
    """No per-event CSV here. Only compute + save the high-level overview."""
    if summary.empty:
        print("No events with enough data. (Check windows/columns)")
        return summary

    warn = summary.groupby("metric", as_index=False)[["warn_pct_drop","warn_sd_drop"]].mean()
    warn = warn.rename(columns={
        "warn_pct_drop": "share_events_flagged_by_pct",
        "warn_sd_drop":  "share_events_flagged_by_sd"
    })
    chg = summary.groupby("metric", as_index=False)[["delta_pre_minus_base","pct_change","sd_change"]].mean()
    overview = warn.merge(chg, on="metric", how="outer")

    overview_file = OUT_DIR / "infection_early_warning_overview.csv"
    overview.to_csv(overview_file, index=False)
    print("Saved overview:", overview_file.name)
    return overview

def plot_histograms(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    for m in METRICS:
        sub = summary.loc[summary["metric"] == m, "delta_pre_minus_base"].dropna()
        if sub.empty:
            continue
        plt.figure(figsize=(6,4))
        sub.plot(kind="hist", bins=30, title=f"Delta before infection — {m} (pre7 - baseline)")
        plt.xlabel("Delta (pre7_mean - baseline_mean)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"hist_delta_{m}_before_infection.png", dpi=150)
        plt.close()

def plot_example_timeline(physio: pd.DataFrame, summary: pd.DataFrame, metric: str) -> None:
    s = summary.loc[summary["metric"] == metric]
    if s.empty:
        return
    r = s.iloc[0]
    pid, e_day = r["STUDY_PRTCPT_ID"], int(r["event_day"])
    p = physio.loc[physio["STUDY_PRTCPT_ID"] == pid, ["DaysFromTransplant", metric]].dropna()
    p = p[(p["DaysFromTransplant"] >= e_day - 40) & (p["DaysFromTransplant"] <= e_day + 7)].sort_values("DaysFromTransplant")
    if p.empty:
        return

    plt.figure(figsize=(7,4))
    plt.plot(p["DaysFromTransplant"], p[metric], marker="o")
    plt.axvline(e_day, color="red", linestyle="--", label="infection day")
    plt.axvspan(e_day + BASE_START, e_day + BASE_END, color="gray", alpha=0.2, label="baseline")
    plt.axvspan(e_day + PRE_START,  e_day + PRE_END,  color="orange", alpha=0.2, label="pre-7")
    plt.title(f"Example timeline — {metric} (ID {pid})")
    plt.xlabel("Days From Transplant")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"example_timeline_{metric}.png", dpi=150)
    plt.close()

# -------------------- Run --------------------
def main() -> None:
    physio, events = load_data()
    summary = compute_summary(physio, events)   # in-memory only
    overview = build_overview(summary)          # only this CSV is saved
    if PLOT:
        plot_histograms(summary)
        if "percent_active" in METRICS and "percent_active" in physio.columns:
            plot_example_timeline(physio, summary, "percent_active")

if __name__ == "__main__":
    main()
