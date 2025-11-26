# Generated script for BPIC11-FormBased - Run 1
# Generated on: 2025-11-18T22:26:39.904004
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run1.csv'
dataset_name = 'bpic11'
case_col = 'Case'
act_col = 'Activity'
ts_col = 'Timestamp'
label_col = 'label'
min_matching_events = 2

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Rename common variants for Case if necessary
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure required columns exist (assume they do as per dataset)
# Convert Timestamp to datetime
df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

# Sort by Case and Timestamp
df = df.sort_values([case_col, ts_col]).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['is_flattened'] = (df.groupby([case_col, ts_col])[case_col].transform('size') >= min_matching_events).astype(int)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_col in df.columns:
    y_true = ((df[label_col].notna()) & (df[label_col] != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'✓' if prec >= 0.6 else '✗'} Precision threshold (>= 0.6) met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 7: Integrity Check
group_sizes = df.groupby([case_col, ts_col]).size()
flattened_groups_count = (group_sizes >= min_matching_events).sum()
total_flattened_events_count = group_sizes[group_sizes >= min_matching_events].sum()
percentage = (total_flattened_events_count / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {flattened_groups_count}")
print(f"Total events marked as flattened: {total_flattened_events_count}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 4: Merge Flattened Activities
merged_rows = []
for (case_val, ts_val), group in flattened_events.groupby([case_col, ts_col]):
    activities = sorted(set(group[act_col].values))  # Unique activities, sorted alphabetically
    merged_activity = ';'.join(activities)
    # Take first row for other columns, including label if present
    first_row = group.iloc[0].copy()
    first_row[act_col] = merged_activity
    merged_rows.append(first_row)

merged_df = pd.DataFrame(merged_rows)

# Drop helper column from normal events
normal_events.drop('is_flattened', axis=1, inplace=True, errors='ignore')

# Drop helper column from merged
merged_df.drop('is_flattened', axis=1, inplace=True, errors='ignore')

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_col, ts_col]).reset_index(drop=True)

# Standardize Timestamp format
final_df[ts_col] = final_df[ts_col].dt.strftime('%Y-%m-%d %H:%M:%S')

# Drop any remaining helper columns (though already dropped)
if 'is_flattened' in final_df.columns:
    final_df.drop('is_flattened', axis=1, inplace=True)

# Step 10: Summary Statistics
orig_events = len(df)
final_events = len(final_df)
num_flattened = len(merged_df)
orig_unique_acts = len(df[act_col].unique())
final_unique_acts = len(final_df[act_col].unique())
reduction = ((orig_events - final_events) / orig_events * 100) if orig_events > 0 else 0
print(f"Total number of events: {final_events}")
print(f"Number of flattened (merged) events detected: {num_flattened}")
print(f"Number of unique activities before vs after merging: {orig_unique_acts} vs {final_unique_acts}")
print(f"Total reduction percentage: {reduction:.2f}%")
print("Sample of up to 10 merged activities:")
merged_samples = final_df[final_df[act_col].str.contains(';', na=False)][act_col].head(10)
for act in merged_samples:
    print(f"  - {act}")
print(f"Output file path: {output_file}")

# Step 9: Save Output (preserve all columns if optional like Resource, Variant exist)
final_df.to_csv(output_file, index=False)

# Required prints
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: {dataset_name}")
print(f"Run 1: Task type: form_based")