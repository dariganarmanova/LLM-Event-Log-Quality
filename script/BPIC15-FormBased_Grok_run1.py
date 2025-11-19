# Generated script for BPIC15-FormBased - Run 1
# Generated on: 2025-11-18T21:41:36.772958
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-FormBased.csv'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run1.csv'
dataset_name = 'bpic15'
task_type = 'form_based'
run_number = 1
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Algorithm parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Step 1: Load CSV
try:
    df = pd.read_csv(input_file)
    print(f"Run {run_number}: Original dataset shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Rename common column variants
column_mapping = {
    'Case ID': 'Case',
    'CaseID': 'Case',
    'case id': 'Case',
    'caseid': 'Case',
    'Activity Name': 'Activity',
    'activity name': 'Activity',
    'event': 'Activity',
    'Event': 'Activity',
    'time': 'Timestamp',
    'timestamp': 'Timestamp',
    'Time': 'Timestamp',
    'Time Stamp': 'Timestamp'
}
df.rename(columns=column_mapping, inplace=True)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# Handle any NaT in Timestamp
if df[timestamp_column].isna().any():
    print("Warning: Some timestamps could not be parsed and set to NaT.")

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['is_flattened'] = df.groupby([case_column, timestamp_column])[case_column].transform('size') >= min_matching_events

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column] != '')).astype(int)
    y_pred = df['is_flattened']
    if y_true.sum() > 0 and y_pred.sum() > 0:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = rec = f1 = 0.0
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    prec = rec = f1 = 0.0
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation.")

prec_threshold_met = "✓" if prec >= 0.6 else "✗"
print(f"{prec_threshold_met} Precision threshold (≥ 0.6) met/not met")

# Step 7: Integrity Check
flattened_mask = df['is_flattened'] == 1
total_flattened_events = flattened_mask.sum()
flattened_groups = df[flattened_mask].groupby([case_column, timestamp_column]).size()
num_flattened_groups = len(flattened_groups)
percentage_flattened = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {num_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Step 3: Preprocess Flattened Groups
normal_events = df[~flattened_mask].copy()
flattened_events = df[flattened_mask].copy()

# Step 4: Merge Flattened Activities
if len(flattened_events) > 0:
    def merge_activity(x):
        return ';'.join(sorted(x.dropna().astype(str)))

    agg_dict = {'Activity': merge_activity}
    other_cols = [col for col in flattened_events.columns if col not in [case_column, timestamp_column, 'is_flattened']]
    for col in other_cols:
        if col != activity_column:
            agg_dict[col] = 'first'

    # Include Case and Timestamp as first (though they are keys)
    agg_dict[case_column] = 'first'
    agg_dict[timestamp_column] = 'first'

    merged_df = flattened_events.groupby([case_column, timestamp_column]).agg(agg_dict).reset_index(drop=True)
else:
    merged_df = pd.DataFrame()

# Step 5: Combine and Sort
if len(merged_df) > 0:
    normal_events_clean = normal_events.drop('is_flattened', axis=1)
    final_df = pd.concat([normal_events_clean, merged_df], ignore_index=True)
else:
    final_df = normal_events.drop('is_flattened', axis=1)

final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Format Timestamp for output
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 8: Fix Events (already done via merging)

# Step 9: Save Output
# Preserve all original columns in output
final_df.to_csv(output_file, index=False)

# Step 10: Summary Statistics
original_events = len(df)
final_events = len(final_df)
num_merged = len(merged_df)
unique_acts_before = df[activity_column].nunique()
unique_acts_after = final_df[activity_column].nunique()
reduction_pct = ((original_events - final_events) / original_events * 100) if original_events > 0 else 0

print(f"Total number of events: {final_events}")
print(f"Number of flattened (merged) events detected: {num_merged}")
print(f"Number of unique activities before: {unique_acts_before}")
print(f"Number of unique activities after: {unique_acts_after}")
print(f"Total reduction percentage: {reduction_pct:.2f}%")
print(f"Output file path: {output_file}")

# Sample merged activities
merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10).tolist()
print("Sample of up to 10 merged activities (;-separated):")
for sample in merged_samples:
    print(sample)

# Required prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")
print(f"This is run #{run_number} of the process")