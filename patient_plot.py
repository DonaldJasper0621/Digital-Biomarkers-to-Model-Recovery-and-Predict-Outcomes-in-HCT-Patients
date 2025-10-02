import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned datasets
physio = pd.read_csv("physiological_dataset_day.csv")
psych  = pd.read_csv("psych_behavioral_dataset_day.csv")
# Pick a few IDs
sample_ids = physio["STUDY_PRTCPT_ID"].unique()[:3]

for pid in sample_ids:
    subset = physio[physio["STUDY_PRTCPT_ID"] == pid]
    plt.plot(subset["DaysFromTransplant"], subset["total_steps"], label=f"ID {pid}")

plt.legend()
plt.title("Steps Over Days (Sample Patients)")
plt.xlabel("Days From Transplant")
plt.ylabel("Steps")
plt.show()

# Mood over days (same patients)
for pid in sample_ids:
    subset = psych[psych["STUDY_PRTCPT_ID"] == pid]
    plt.plot(subset["DaysFromTransplant"], subset["MOOD"], label=f"ID {pid}")

plt.legend()
plt.title("Mood Over Days (Sample Patients)")
plt.xlabel("Days From Transplant")
plt.ylabel("Mood Score")
plt.show()

# Average mood stratified
psych_strat = psych.groupby(["role","gender","transplant_type"])["MOOD"].mean().reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x="gender", y="MOOD", hue="role", data=psych_strat)
plt.title("Average Mood by Gender and Role")
plt.show()

