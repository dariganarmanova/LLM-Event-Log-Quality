# Generated script for Credit-Formbased - Run 2
# Generated on: 2025-11-13T16:19:23.620093
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
input_file = 'data/credit/Credit-Formbased.csv'
input_directory = 'data/credit'
dataset_name = 'credit'
output_suffix = '_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Time'
label_column = 'label'

df = pd.read_csv(input_file)

# Rename common variants
df = df.rename(columns={'CaseID': case_column})

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='%H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df = df.sort_values(by=[case_column, timestamp_column])

# Create group key: combination of Case and Timestamp values
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].dt.strftime('%H:%M:%S')

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size()

# Mark is_flattened = 1 if group size ≥ 2; otherwise 0
df['is_flattened'] = df['group_key'].map(group_counts) >= min_matching_events

# Split dataset: normal_events and flattened_events
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
flattened_events = flattened_events.groupby('group_key')[activity_column].apply(lambda x: ';'.join(sorted(x))).reset_index()

# Keep the first label (if present)
flattened_events = flattened_events.merge(normal_events[['group_key', label_column]], on='group_key', how='left').fillna('')

# Create a new DataFrame with merged records
merged_df = pd.concat([normal_events, flattened_events])

# Concatenate normal_events with merged flattened_events
merged_df = pd.concat([normal_events, flattened_events])

# Sort final DataFrame by Case and Timestamp
merged_df = merged_df.sort_values(by=[case_column, timestamp_column])

# Drop helper columns (group_key, is_flattened)
merged_df = merged_df.drop(columns=['group_key', 'is_flattened'])

# Calculate detection metrics (if label column exists)
if label_column in merged_df.columns:
    y_true = merged_df[label_column].notnull().astype(int)
    y_pred = merged_df['is_flattened'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met: {'✓' if precision >= 0.6 else '✗'}")
else:
    print("No labels available for metric calculation.")

# Integrity check
flattened_groups = merged_df['group_key'].nunique()
flattened_events_count = merged_df['is_flattened'].sum()
percentage_flattened = (flattened_events_count / merged_df.shape[0]) * 100
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {flattened_events_count}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Fix events
merged_df[activity_column] = merged_df[activity_column].apply(lambda x: re.sub(activity_suffix_pattern, '', x))

# Save output CSV
output_path = input_directory + '/' + dataset_name + output_suffix
merged_df.to_csv(output_path, index=False)

# Print summary
print(f"Run 2: Processed dataset saved to: {output_path}")
print(f"Run 2: Final dataset shape: {merged_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: form_based")