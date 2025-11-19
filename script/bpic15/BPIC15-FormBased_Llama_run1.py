# Generated script for BPIC15-FormBased - Run 1
# Generated on: 2025-11-13T14:32:06.897526
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
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
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
output_suffix = '_form_based_cleaned_run1'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Time'
label_column = 'label'

# Load the data
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants (e.g., `CaseID` → `Case`)
df = df.rename(columns={'CaseID': case_column})

# Convert the `Timestamp` column to datetime and standardize as `YYYY-MM-DD HH:MM:SS`
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='%H:%M:%S')

# Sort by `Case` and `Timestamp` to maintain chronological order
df = df.sort_values(by=[case_column, timestamp_column])

# Create `group_key`: combination of `Case` and `Timestamp` values
df['group_key'] = df.apply(lambda row: f"{row[case_column]}_{row[timestamp_column]}", axis=1)

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size()

# Mark `is_flattened = 1` if group size ≥ 2; otherwise 0
df['is_flattened'] = df['group_key'].apply(lambda x: 1 if group_counts[x] >= min_matching_events else 0)

# Split dataset:
# `normal_events`: `is_flattened == 0`
# `flattened_events`: `is_flattened == 1`
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Group flattened events by `(Case, Timestamp)`
flattened_groups = flattened_events.groupby(['Case', timestamp_column])

# For each group:
# Merge all `Activity` values alphabetically using `;` as separator.
# Keep the *first** `label` (if present).
merged_activities = []
for name, group in flattened_groups:
    activities = sorted(group[activity_column].unique())
    merged_activity = ';'.join(activities)
    label = group[label_column].iloc[0] if not group[label_column].isnull().all() else None
    merged_activities.append({'Case': name[0], 'Time': name[1], 'Activity': merged_activity, 'label': label})

# Create a new DataFrame with merged records
merged_events = pd.DataFrame(merged_activities)

# Concatenate `normal_events` with merged `flattened_events`
final_df = pd.concat([normal_events, merged_events])

# Sort final DataFrame by `Case` and `Timestamp`
final_df = final_df.sort_values(by=[case_column, timestamp_column])

# Drop helper columns (`group_key`, `is_flattened`)
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# If `label` column exists:
if label_column in final_df.columns:
    # Define `y_true`: 1 if `label` not null/empty, else 0.
    y_true = final_df[label_column].notnull().astype(int)
    
    # Define `y_pred`: value of `is_flattened`.
    y_pred = final_df['is_flattened']
    
    # Compute precision, recall, and F1-score using sklearn metrics.
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print metrics
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Check precision threshold
    if precision >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")

# If no label column exists:
else:
    print("No labels available for metric calculation.")
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")

# Integrity check
flattened_groups_count = len(flattened_groups.groups)
flattened_events_count = len(flattened_events)
flattened_percentage = (flattened_events_count / len(df)) * 100

print(f"Total flattened groups detected: {flattened_groups_count}")
print(f"Total events marked as flattened: {flattened_events_count}")
print(f"Percentage of flattened events: {flattened_percentage:.2f}%")

# Save output
output_path = input_directory + '/' + dataset_name + output_suffix + '.csv'
final_df.to_csv(output_path, index=False)

# Summary statistics
print(f"Total number of events: {len(final_df)}")
print(f"Number of flattened (merged) events detected: {flattened_events_count}")
print(f"Number of unique activities before vs after merging: {len(df[activity_column].unique())} vs {len(final_df[activity_column].unique())}")
print(f"Total reduction percentage: {(1 - (flattened_events_count / len(df))) * 100:.2f}%")
print(f"Output file path: {output_path}")

# Print sample of up to 10 merged activities (`;` separated)
merged_activities_sample = final_df['Activity'].head(10).tolist()
print(f"Merged activities sample: {', '.join(merged_activities_sample)}")