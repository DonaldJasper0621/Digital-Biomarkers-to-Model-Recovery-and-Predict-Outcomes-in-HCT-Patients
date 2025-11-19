# Digital Biomarkers to Model Recovery & Predict Outcomes in HCT Patients

> Connect passively collected wearables data (steps, sleep, HR) with clinical outcomes after hematopoietic cell transplant (HCT). Explore whether behavioral/physiological signals can model recovery trajectories and flag adverse events (e.g., infections, readmission).

## ⚙️ Quick start

```bash
# 1) Create and activate a virtual environment (Python ≥3.10)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install dependencies (minimal)
pip install -r requirements.txt  # if present
# or minimally:
pip install pandas numpy matplotlib scipy

# 3) Place data CSVs in the repo root (already included here):
#   - physiological_dataset_day.csv
#   - psych_behavioral_dataset_day.csv
#   - events_infections.csv
#   - events_outcomes.csv

# 4) Run the pipeline pieces (examples)
python build_two_datasets_simple.py
python check_heads_and_rationale.py
python recovery_patients_with_caregiver_baseline.py
python early_warning_signs.py
```

> **Tip:** All scripts read CSVs from the project root and write any outputs to a local subfolder (e.g., `early_warning_outputs/`).

## 🗂️ Repository layout

```
.
├── build_two_datasets_simple.py          # Build/clean derived daily datasets used by downstream scripts
├── check_heads_and_rationale.py          # Sanity checks on headers; prints rationale and column expectations
├── recovery_patients_with_caregiver_baseline.py
│   └─ Compare post‑transplant patient recovery to caregiver “baseline” on daily metrics (steps, sleep, HR, mood)
├── early_warning_signs.py                # Event‑centered windows around infections/readmissions to detect pre‑event signal drops
├── events_infections.csv                 # Event log (dates/IDs) for infection episodes
├── events_outcomes.csv                   # Event log for other clinical outcomes (e.g., readmission)
├── physiological_dataset_day.csv         # Daily wearables/physiology (steps, HR, sleep, etc.)
├── psych_behavioral_dataset_day.csv      # Daily mood/behavioral measures (e.g., MOOD, PROMIS‑like scores)
├── Initial_data/                         # (Optional) scratch or staging files
├── Trajectory/                           # (Optional) figures / trajectory artifacts
├── data_unused/                          # (Optional) archival inputs not used in v1
├── early_warning_outputs/                # Generated plots/tables from early‑warning analysis
├── Rationale.txt                         # Notes for design decisions & column checks
└── .gitignore
```

## 📚 Data fields (high‑level)

* **physiological_dataset_day.csv** — expected columns include `STUDY_PRTCPT_ID`, `DaysFromTransplant`, daily aggregates such as `total_steps`, `sleep_duration`, `mean_hr`, and optional derivatives like `percent_active`, `sleep_efficiency` (if `ASLEEP_MIN` and `INBED_VALUE` exist).
* **psych_behavioral_dataset_day.csv** — expected columns include `STUDY_PRTCPT_ID`, `DaysFromTransplant`, and psych/behavioral fields such as `MOOD`.
* **events_infections.csv / events_outcomes.csv** — event‑level rows keyed by participant and an event date for aligning windows.

## 🧪 What each script does

### 1) `build_two_datasets_simple.py`

* Loads daily **physiological** and **psych/behavioral** CSVs.
* Derives helpful features when possible (e.g., `sleep_efficiency = ASLEEP_MIN / INBED_VALUE`).
* Produces clean daily tables keyed by `STUDY_PRTCPT_ID` × `DaysFromTransplant` for downstream analyses.

### 2) `check_heads_and_rationale.py`

* Prints sample heads and column checks so you can confirm data shape and naming across files.
* Echoes/reads from `Rationale.txt` to document why specific fields are required or skipped.

### 3) `recovery_patients_with_caregiver_baseline.py`

* Focus: **Recovery vs. Baseline**.
* Conceptually compares patients’ post‑transplant daily metrics versus their matched caregiver’s pre‑defined baseline period to quantify recovery gaps.
* Useful to visualize/quantify how quickly a patient returns to “caregiver‑like” activity/sleep levels.

### 4) `early_warning_signs.py`

* Focus: **Event‑centered early warning** for infections/readmissions.
* Uses two windows per participant around each event:

  * **Baseline window:** typically −30 to −14 days relative to the event date.
  * **Pre‑event window:** typically −7 to −1 days.
* Compares changes in metrics (e.g., `total_steps`, `percent_active`, `mean_hr`, `sleep_duration`).
* Flags notable drops (e.g., % change or z/sd change thresholds) and can export plots/tables under `early_warning_outputs/`.

> These time windows and metrics are configurable inside the script via constants near the top (see the `Windows` dataclass and `METRICS`).

## 🔍 Typical questions this repo answers

* **Do daily steps, HR, and sleep change *before* an infection?**
* **How do patients’ daily patterns compare to caregiver baselines?**
* **Which metrics are most sensitive/stable for early warnings?**

## 🧰 Dependencies

* Python ≥ 3.10
* `pandas`, `numpy`, `matplotlib`, `scipy`

*(If you formalize a `requirements.txt`, list exact versions you tested with; example: pandas ≥2.2, numpy ≥1.26, matplotlib ≥3.8, scipy ≥1.12.)*

## 🗜️ Reproducibility notes

* All scripts expect the four CSVs in the project root, UTF‑8 encoded, with consistent `STUDY_PRTCPT_ID` and `DaysFromTransplant` keys.
* Minor naming differences (e.g., `Sara` vs. `Sarah`) across sources are **not** reconciled here; matching is done by the keys present in the daily tables.
* If some optional columns are missing, scripts should skip those derived features gracefully.

## 📈 Outputs

* **Early‑warning tables/figures** in `early_warning_outputs/` (per‑event metric trends & marked drops).
* **Printed sanity checks** to the console for column heads and basic statistics.

## 🧭 Roadmap (ideas)

* Add `argparse` CLIs (e.g., `--baseline -30 -14 --pre -7 -1 --metrics steps,hr,sleep`).
* Add unit tests for feature derivation and windowing.
* Provide example notebooks for EDA and model training.
* Integrate lightweight models (e.g., logistic regression) for event prediction.

## 🔒 Ethics & privacy

* This project uses de‑identified, IRB‑approved datasets. When extending, ensure all data handling respects HIPAA/PHI rules and your local IRB guidance.

## 📝 Citation

If you use this repo, please cite it as:

> Su, D. J. (2025). *Digital Biomarkers to Model Recovery & Predict Outcomes in HCT Patients* (Version 1.0) [Computer software]. GitHub: DonaldJasper0621.

---

**Maintainer:** Donald Jasper Su  ·  Issues and PRs welcome!
