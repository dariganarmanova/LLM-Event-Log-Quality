# Generated script for Credit-Formbased - Run 1
# Generated on: 2025-11-18T19:22:34.036363
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
output_suffix = '_form_based_cleaned_run1'  # Adjusted for specific output
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Rename common column variants
column_mapping = {
    'Case ID': 'Case',
    'CaseID': 'Case',
    'case': 'Case',
    'Activity': 'Activity',
    'activity': 'Activity',
    'event': 'Activity',
    'Timestamp': 'Timestamp',
    'timestamp': 'Timestamp',
    'time': 'Timestamp',
    'label': 'label'
}
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Flattened Events
group_sizes = df.groupby([case_column, timestamp_column]).size()
df = df.join(group_sizes.rename('group_size'), on=[case_column, timestamp_column], how='left')
df['is_flattened'] = (df['group_size'] >= min_matching_events).astype(int)
df.drop('group_size', axis=1, inplace=True)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
label_exists = label_column in df.columns
if label_exists:
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    met = "✓" if prec >= 0.6 else "✗"
    status = "met" if prec >= 0.6 else "not met"
    print(f"{met} Precision threshold (≥ 0.6) {status}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 7: Integrity Check
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)
flattened_groups = df[df['is_flattened'] == 1]['group_key'].nunique()
flattened_events_count = df['is_flattened'].sum()
total_events = len(df)
percentage = (flattened_events_count / total_events * 100) if total_events > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {flattened_events_count}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 4: Merge Flattened Activities
def merge_group(g):
    data = {}
    for col in g.columns:
        if col == activity_column:
            acts = sorted(g[col].tolist())
            data[col] = ';'.join(acts)
        elif col in ['group_key', 'is_flattened']:
            continue
        else:
            data[col] = g[col].iloc[0]
    return pd.Series(data)

if len(flattened_events) > 0:
    merged_df = flattened_events.groupby('group_key').apply(merge_group).reset_index(drop=True)
else:
    # If no flattened events, create empty df with structure
    merged_df = pd.DataFrame(columns=[col for col in df.columns if col not in ['group_key', 'is_flattened']])

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)
# Drop helper columns
helper_cols = ['group_key', 'is_flattened']
final_df = final_df.drop(columns=[c for c in helper_cols if c in final_df.columns])

# Preserve original columns (including optional like Variant, Resource if present)
original_cols = [col for col in df.columns if col not in ['group_key', 'is_flattened']]
final_df = final_df[[col for col in original_cols if col in final_df.columns]]

# Format Timestamp
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 10: Summary Statistics
print(f"Total number of events: {len(final_df)}")
num_merged = len(merged_df)
print(f"Number of flattened (merged) events detected: {num_merged}")
unique_acts_before = df[activity_column].nunique()
unique_acts_after = final_df[activity_column].nunique()
print(f"Number of unique activities before vs after merging: {unique_acts_before} vs {unique_acts_after}")
reduction = ((len(df) - len(final_df)) / len(df) * 100) if len(df) > 0 else 0
print(f"Total reduction percentage: {reduction:.2f}%")
print(f"Output file path: data/credit/credit_form_based_cleaned_run1.csv")

# Sample of up to 10 merged activities
merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10).tolist()
print("Sample of up to 10 merged activities (`;` separated):")
for act in merged_samples:
    print(f"  - {act}")

# Step 9: Save Output
output_path = 'data/credit/credit_form_based_cleaned_run1.csv'
final_df.to_csv(output_path, index=False)

# REQUIRED: Print summary
print(f"Run 1: Processed dataset saved to: data/credit/credit_form_based_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: form_based")