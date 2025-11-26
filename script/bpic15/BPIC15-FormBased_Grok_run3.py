# Generated script for BPIC15-FormBased - Run 3
# Generated on: 2025-11-18T21:43:01.606838
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Input configuration
input_file = 'data/bpic15/BPIC15-FormBased.csv'
dataset_name = 'bpic15'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Step 1: Load CSV
df = pd.read_csv(input_file)

# Rename common variants
column_mapping = {
    'Case ID': 'Case',
    'case id': 'Case',
    'Activity ID': 'Activity',
    'Complete Timestamp': 'Timestamp',
    'time:timestamp': 'Timestamp',
    'timestamp': 'Timestamp'
}
for old, new in column_mapping.items():
    if old in df.columns:
        df[new] = df[old]
        df = df.drop(old, axis=1)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
df = df.dropna(subset=[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['group_size'] = df.groupby([case_column, timestamp_column])[case_column].transform('size')
df['is_flattened'] = (df['group_size'] >= min_matching_events).astype(int)

if 'Resource' in df.columns:
    df['res_nunique'] = df.groupby([case_column, timestamp_column])['Resource'].transform('nunique')
    if require_same_resource:
        df['is_flattened'] = df['is_flattened'] & (df['res_nunique'] == 1).astype(int)
    else:
        df['is_flattened'] = df['is_flattened'] & (df['res_nunique'] <= max_mismatches + 1).astype(int)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'✓' if prec >= 0.6 else '✗'} Precision threshold (>= 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 7: Integrity Check
flattened_mask = df['is_flattened'] == 1
flattened_groups = df[flattened_mask].groupby([case_column, timestamp_column]).size().shape[0]
total_flattened_events = flattened_mask.sum()
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 4: Merge Flattened Activities
def merge_group(g):
    activities = sorted(g[activity_column].dropna().unique(), key=str.lower if not case_sensitive else None)
    merged_activity = ';'.join(activities)
    row = {
        case_column: g[case_column].iloc[0],
        timestamp_column: g[timestamp_column].iloc[0],
        activity_column: merged_activity
    }
    exclude_cols = [case_column, timestamp_column, activity_column, 'group_size', 'is_flattened', 'res_nunique']
    for col in df.columns:
        if col not in exclude_cols:
            row[col] = g[col].iloc[0]
    return pd.Series(row)

if len(flattened_events) > 0:
    merged_df = flattened_events.groupby([case_column, timestamp_column]).apply(merge_group).reset_index(drop=True)
else:
    merged_df = pd.DataFrame(columns=df.columns)

# Step 5: Combine and Sort
exclude_cols = ['group_size', 'is_flattened', 'res_nunique']
normal_events_clean = normal_events.drop(columns=[col for col in exclude_cols if col in normal_events.columns], errors='ignore')
final_df = pd.concat([normal_events_clean, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 10: Summary Statistics
original_events = len(df)
final_events = len(final_df)
unique_acts_before = df[activity_column].nunique()
unique_acts_after = final_df[activity_column].nunique()
reduction_pct = ((original_events - final_events) / original_events * 100) if original_events > 0 else 0
print(f"Total number of events before: {original_events}")
print(f"Number of flattened (merged) events detected: {flattened_groups}")
print(f"Number of unique activities before: {unique_acts_before}")
print(f"Number of unique activities after: {unique_acts_after}")
print(f"Total reduction percentage: {reduction_pct:.2f}%")

merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10).tolist()
print("Sample of up to 10 merged activities (;-separated):")
for sample in merged_samples:
    print(sample)

# Step 9: Save Output
# Standardize timestamp format
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
final_df.to_csv(output_file, index=False)
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 3: Processed dataset saved to: data/bpic15/bpic15_form_based_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: form_based")
print("Run 3: Summary - Form-based flattening detected and merged simultaneous events in the same case and timestamp.")