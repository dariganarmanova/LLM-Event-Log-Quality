# Generated script for Credit-Formbased - Run 2
# Generated on: 2025-11-18T19:23:16.791336
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
input_file = 'data/credit/Credit-Formbased.csv'
dataset_name = 'credit'
output_suffix = '_form_based_cleaned_run2'  # Adjusted for specific output
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Handle column renaming if necessary (common variants)
column_mapping = {
    'Case ID': case_column,
    'case': case_column,
    'case_id': case_column,
    'Activity': activity_column,
    'activity': activity_column,
    'event': activity_column,
    'Event': activity_column,
    'Timestamp': timestamp_column,
    'timestamp': timestamp_column,
    'time': timestamp_column,
    'Time': timestamp_column
}
df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert Timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
df[timestamp_column] = df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Flattened Events
group_key = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)
counts = df.groupby(group_key).size()
df['is_flattened'] = df[group_key].map(counts).ge(min_matching_events).astype(int)

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy().drop('is_flattened', axis=1)
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column] != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    status = "✓" if prec >= 0.6 else "✗"
    met = "met" if prec >= 0.6 else "not met"
    print(f"{status} Precision threshold (>= 0.6) {met}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 7: Integrity Check
flattened_groups = flattened_events[[case_column, timestamp_column]].drop_duplicates().shape[0]
total_flattened_events = len(flattened_events)
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 4: Merge Flattened Activities
merged_flattened = []
for name, group in flattened_events.groupby([case_column, timestamp_column]):
    acts = [a for a in group[activity_column].dropna().tolist() if pd.notna(a)]
    unique_acts = list(set(acts))
    if case_sensitive:
        sorted_acts = sorted(unique_acts)
    else:
        sorted_acts = sorted(unique_acts, key=str.lower)
    merged_act = ';'.join(sorted_acts)
    merged_row = group.iloc[0].copy()
    merged_row[activity_column] = merged_act
    # Preserve other columns from first row (e.g., Resource, Variant if present)
    merged_flattened.append(merged_row)

merged_flattened = pd.DataFrame(merged_flattened)

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 8: Fix Events (already done via merging)

# Step 10: Summary Statistics
total_events_after = len(final_df)
num_flattened_merged = len(merged_flattened)
unique_acts_before = df[activity_column].nunique()
unique_acts_after = final_df[activity_column].nunique()
original_rows = len(df)
reduction = ((original_rows - total_events_after) / original_rows * 100) if original_rows > 0 else 0
print(f"Total number of events: {total_events_after}")
print(f"Number of flattened (merged) events detected: {num_flattened_merged}")
print(f"Number of unique activities before: {unique_acts_before}")
print(f"Number of unique activities after: {unique_acts_after}")
print(f"Total reduction percentage: {reduction:.2f}%")

# Sample of up to 10 merged activities
merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10).tolist()
print("Sample merged activities (up to 10):")
for act in merged_samples:
    print(f"  - {act}")

# Step 9: Save Output
# Include all columns to preserve optional ones like Variant, Resource
output_path = 'data/credit/credit_form_based_cleaned_run2.csv'
final_df.to_csv(output_path, index=False)
print(f"Output file path: {output_path}")

# Required prints
print(f"Run 2: Processed dataset saved to: data/credit/credit_form_based_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: form_based")
print("Run 2: Summary - Form-based flattening detected and merged events with identical timestamps per case.")