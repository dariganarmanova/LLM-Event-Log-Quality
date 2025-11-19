# Generated script for Credit-Formbased - Run 3
# Generated on: 2025-11-13T16:19:25.638546
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
input_file = 'data/credit/Credit-Formbased.csv'
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants (e.g., `CaseID` → `Case`)
df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Convert the `Timestamp` column to datetime and standardize as `YYYY-MM-DD HH:MM:SS`
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')

# Sort by `Case` and `Timestamp` to maintain chronological order
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Create `group_key`: combination of `Case` and `Timestamp` values
df['group_key'] = df['Case'] + '_' + df['Timestamp'].dt.strftime('%H:%M:%S')

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size()

# Mark `is_flattened = 1` if group size ≥ 2; otherwise 0
df['is_flattened'] = df['group_key'].map(group_counts) >= min_matching_events

# Split dataset:
# * `normal_events`: `is_flattened == 0`
# * `flattened_events`: `is_flattened == 1`
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Group flattened events by `(Case, Timestamp)`
flattened_groups = flattened_events.groupby(['Case', 'Timestamp'])

# For each group:
# * Merge all `Activity` values alphabetically using `;` as separator.
# * Keep the *first** `label` (if present).
merged_activities = []
for name, group in flattened_groups:
    activities = sorted(group['Activity'].unique().tolist())
    merged_activity = ';'.join(activities)
    merged_activities.append(merged_activity)
    if 'label' in group.columns and not group['label'].isnull().all():
        merged_label = group['label'].iloc[0]
    else:
        merged_label = None
    merged_group = pd.DataFrame({
        'Case': [name[0]],
        'Timestamp': [name[1]],
        'Activity': [merged_activity],
        'label': [merged_label]
    })
    merged_events = pd.concat([merged_group, normal_events[normal_events['Case'] == name[0]]])

# Create a new DataFrame with merged records
merged_df = pd.concat([merged_events, flattened_events[~flattened_events['group_key'].isin(flattened_groups.groups.keys())]])

# Concatenate `normal_events` with merged `flattened_events`
final_df = pd.concat([normal_events, merged_df])

# Sort final DataFrame by `Case` and `Timestamp`
final_df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Drop helper columns (`group_key`, `is_flattened`)
final_df.drop(columns=['group_key', 'is_flattened'], inplace=True)

# If `label` column exists:
if 'label' in final_df.columns:
    # Define `y_true`: 1 if `label` not null/empty, else 0.
    y_true = final_df['label'].notnull().astype(int)
    # Define `y_pred`: value of `is_flattened`.
    y_pred = final_df['is_flattened'].astype(int)
    # Compute precision, recall, and F1-score using sklearn metrics.
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
    print(f"{'✓' if precision >= 0.6 else '✗'}")

# Count and print:
# * Total flattened groups detected.
# * Total events marked as flattened.
# * Percentage of flattened events.
flattened_groups_count = len(flattened_groups.groups.keys())
flattened_events_count = flattened_events.shape[0]
percentage_flattened = (flattened_events_count / final_df.shape[0]) * 100
print(f"Total flattened groups detected: {flattened_groups_count}")
print(f"Total events marked as flattened: {flattened_events_count}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Replace original flattened events with merged results.
final_df.loc[final_df['is_flattened'] == 1, 'Activity'] = merged_activities

# Select columns: `Case`, `Activity`, `Timestamp`, and `label` (if present).
final_df = final_df[['Case', 'Activity', 'Timestamp', 'label']]

# Ensure timestamp format `YYYY-MM-DD HH:MM:SS`.
final_df['Timestamp'] = final_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save output CSV as:
output_path = 'data/credit/credit_form_based_cleaned_run3.csv'
final_df.to_csv(output_path, index=False)

# Print summary
print(f"Run 3: Processed dataset saved to: {output_path}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: form_based")

# Print the path where the data was saved
print(f"Data saved to: {output_path}")

# Print the shape of the final dataset
print(f"Final dataset shape: {final_df.shape}")