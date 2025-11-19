# Generated script for BPIC15-FormBased - Run 3
# Generated on: 2025-11-13T14:32:12.539102
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
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
input_file = 'data/bpic15/BPIC15-FormBased.csv'
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants (e.g., `CaseID` → `Case`)
df = df.rename(columns={'CaseID': 'Case'})

# Convert the `Timestamp` column to datetime and standardize as `YYYY-MM-DD HH:MM:SS`
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')

# Sort by `Case` and `Timestamp` to maintain chronological order
df = df.sort_values(by=['Case', 'Timestamp'])

# Create `group_key`: combination of `Case` and `Timestamp` values
df['group_key'] = df['Case'] + '_' + df['Timestamp'].dt.strftime('%H:%M:%S')

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size()

# Mark `is_flattened = 1` if group size ≥ 2; otherwise 0
df['is_flattened'] = np.where(group_counts > 1, 1, 0)

# Split dataset: `normal_events` and `flattened_events`
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Group flattened events by `(Case, Timestamp)`
flattened_groups = flattened_events.groupby(['Case', 'Timestamp'])

# For each group: merge all `Activity` values alphabetically using `;` as separator
merged_activities = flattened_groups['Activity'].apply(lambda x: ';'.join(sorted(x.unique())))

# Keep the *first** `label` (if present)
merged_labels = flattened_groups['label'].first()

# Create a new DataFrame with merged records
merged_df = pd.DataFrame({'Case': flattened_groups.ngroup(),
                          'Timestamp': flattened_groups.groups.keys(),
                          'Activity': merged_activities,
                          'label': merged_labels})

# Concatenate `normal_events` with merged `flattened_events`
final_df = pd.concat([normal_events, merged_df])

# Sort final DataFrame by `Case` and `Timestamp`
final_df = final_df.sort_values(by=['Case', 'Timestamp'])

# Drop helper columns (`group_key`, `is_flattened`)
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# If `label` column exists:
if 'label' in final_df.columns:
    # Define `y_true`: 1 if `label` not null/empty, else 0
    y_true = final_df['label'].notnull().astype(int)

    # Define `y_pred`: value of `is_flattened`
    y_pred = final_df['is_flattened']

    # Compute precision, recall, and F1-score using sklearn metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print metrics
    print('=== Detection Performance Metrics ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'✓/✗ Precision threshold (≥ 0.6) met/not met')

    # Check if precision threshold is met
    if precision >= 0.6:
        print('✓')
    else:
        print('✗')

# If no label column exists:
else:
    print('No labels available for metric calculation.')

# Count and print:
total_flattened_groups = len(flattened_groups.groups)
total_flattened_events = len(flattened_events)
percentage_flattened_events = (total_flattened_events / len(df)) * 100

# Print integrity check results
print(f'Total flattened groups detected: {total_flattened_groups}')
print(f'Total events marked as flattened: {total_flattened_events}')
print(f'Percentage of flattened events: {percentage_flattened_events:.2f}%')

# Replace original flattened events with merged results
final_df = pd.concat([normal_events, merged_df])

# Save output CSV
output_path = 'data/bpic15/bpic15_form_based_cleaned_run3.csv'
final_df.to_csv(output_path, index=False)

# Print summary statistics
total_events = len(final_df)
unique_activities_before = len(df['Activity'].unique())
unique_activities_after = len(final_df['Activity'].unique())
reduction_percentage = ((unique_activities_before - unique_activities_after) / unique_activities_before) * 100

# Print summary statistics
print(f'Total number of events: {total_events}')
print(f'Number of flattened (merged) events detected: {total_flattened_events}')
print(f'Number of unique activities before merging: {unique_activities_before}')
print(f'Number of unique activities after merging: {unique_activities_after}')
print(f'Total reduction percentage: {reduction_percentage:.2f}%')
print(f'Output file path: {output_path}')

# Print sample of up to 10 merged activities
print('Sample of merged activities:')
print(final_df['Activity'].head(10))