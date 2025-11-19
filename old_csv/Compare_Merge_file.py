import pandas as pd

# -------- File paths --------
old_physio = "physiological_dataset_day.csv"
new_physio = "NEW_physiological_dataset_day.csv"

old_psych = "psych_behavioral_dataset_day.csv"
new_psych = "NEW_psych_behavioral_dataset_day.csv"

# -------- Helper function --------
def compare_csvs(file_old, file_new, key_columns=None):
    print(f"\n=== Comparing: {file_old}  vs  {file_new} ===")

    # Read both
    df_old = pd.read_csv(file_old)
    df_new = pd.read_csv(file_new)

    # Show basic shape comparison
    print(f"Old shape: {df_old.shape}, New shape: {df_new.shape}")

    # Compare columns
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)
    if old_cols != new_cols:
        print("\nColumn differences:")
        print("  Only in old:", old_cols - new_cols)
        print("  Only in new:", new_cols - old_cols)
    else:
        print("\n✅ Columns identical.")

    # Compare row-by-row if same shape
    if df_old.shape == df_new.shape and set(df_old.columns) == set(df_new.columns):
        diff = (df_old != df_new).sum().sum()
        if diff == 0:
            print("✅ All values identical.")
        else:
            print(f"⚠️  Found {diff} different cell(s). Showing first few differences:")
            # Show example rows that differ
            diff_rows = df_old.ne(df_new).any(axis=1)
            display_cols = list(df_old.columns[:6])  # show first few cols for readability
            print(pd.concat([df_old[diff_rows][display_cols].head(),
                             df_new[diff_rows][display_cols].head()],
                            keys=["old", "new"]))
    else:
        print("\n⚠️ Shapes or columns differ — skipping cell-by-cell check.")

# -------- Run comparisons --------
compare_csvs(old_physio, new_physio)
compare_csvs(old_psych, new_psych)
