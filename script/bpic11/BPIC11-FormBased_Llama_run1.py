# Generated script for BPIC11-FormBased - Run 1
# Generated on: 2025-11-13T11:17:45.490615
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

# Load the data
input_file = 'data/bpic11/BPIC11-FormBased.csv'
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants
df = df.rename(columns={'CaseID': 'Case'})

# Convert the `Timestamp` column to datetime and standardize as `YYYY-MM-DD HH:MM:SS`
df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by `Case` and `Timestamp` to maintain chronological order
df = df.sort_values(by=['Case', 'Timestamp'])

# Create `group_key`: combination of `Case` and `Timestamp` values
df['group_key'] = df['Case'] + '_' + df['Timestamp'].astype(str)

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size().reset_index(name='count')

# Mark `is_flattened = 1` if group size ≥ 2; otherwise 0
group_counts['is_flattened'] = group_counts['count'].apply(lambda x: 1 if x >= min_matching_events else 0)

# Merge with original dataframe
df = pd.merge(df, group_counts, on='group_key')

# Split dataset:
# * `normal_events`: `is_flattened == 0`
# * `flattened_events`: `is_flattened == 1`
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Group flattened events by `(Case, Timestamp)`
flattened_events = flattened_events.groupby(['Case', 'Timestamp']).agg({
    'Activity': lambda x: ';'.join(sorted(x.unique())),
    'label': 'first'
}).reset_index()

# Create a new DataFrame with merged records
merged_events = pd.concat([normal_events, flattened_events])

# Concatenate `normal_events` with merged `flattened_events`
merged_events = pd.concat([normal_events, flattened_events])

# Sort final DataFrame by `Case` and `Timestamp`
merged_events = merged_events.sort_values(by=['Case', 'Timestamp'])

# Drop helper columns (`group_key`, `is_flattened`)
merged_events = merged_events.drop(columns=['group_key', 'is_flattened'])

# If `label` column exists:
if 'label' in merged_events.columns:
    # Define `y_true`: 1 if `label` not null/empty, else 0
    y_true = merged_events['label'].notnull().astype(int)

    # Define `y_pred`: value of `is_flattened`
    y_pred = merged_events['is_flattened']

    # Compute precision, recall, and F1-score using sklearn metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print metrics
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")

    # Check if precision threshold is met
    if precision >= 0.6:
        print("✓")
    else:
        print("✗")

else:
    # Print metrics as 0.0000
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Count and print:
# * Total flattened groups detected.
# * Total events marked as flattened.
# * Percentage of flattened events
total_flattened_groups = flattened_events.shape[0]
total_events_marked_as_flattened = flattened_events.shape[0]
percentage_flattened_events = (total_events_marked_as_flattened / merged_events.shape[0]) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_events_marked_as_flattened}")
print(f"Percentage of flattened events: {percentage_flattened_events:.2f}%")

# Save output CSV
output_path = 'data/bpic11/bpic11_form_based_cleaned_run1.csv'
merged_events.to_csv(output_path, index=False)

# Print summary
print(f"Run 1: Processed dataset saved to: {output_path}")
print(f"Run 1: Final dataset shape: {merged_events.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: form_based")

# Print the path where the data was saved
print(f"Run 1: Data saved to: {output_path}")

# Print the shape of the final dataset
print(f"Run 1: Final dataset shape: {merged_events.shape}")

# Print a summary of what was processed for run 1
print(f"Run 1: Processed {merged_events.shape[0]} events")

# Print the number of unique activities before vs after merging
unique_activities_before = df['Activity'].nunique()
unique_activities_after = merged_events['Activity'].nunique()
print(f"Run 1: Unique activities before merging: {unique_activities_before}")
print(f"Run 1: Unique activities after merging: {unique_activities_after}")

# Print the total reduction percentage
total_reduction_percentage = ((unique_activities_before - unique_activities_after) / unique_activities_before) * 100
print(f"Run 1: Total reduction percentage: {total_reduction_percentage:.2f}%")

# Print a sample of up to 10 merged activities (`;` separated)
sample_merged_activities = merged_events['Activity'].sample(n=10).tolist()
print(f"Run 1: Sample of merged activities: {'; '.join(sample_merged_activities)}")