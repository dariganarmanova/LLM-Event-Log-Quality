# Generated script for Pub-Formbased - Run 1
# Generated on: 2025-11-14T13:29:18.771217
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
input_file = 'data/pub/Pub-Formbased.csv'
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants (e.g., `CaseID` → `Case`)
df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Convert the `Timestamp` column to datetime and standardize as `YYYY-MM-DD HH:MM:SS`
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by `Case` and `Timestamp` to maintain chronological order
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Create `group_key`: combination of `Case` and `Timestamp` values
df['group_key'] = df['Case'] + '_' + df['Timestamp']

# Count occurrences per group using groupby
group_counts = df.groupby('group_key').size()

# Mark `is_flattened = 1` if group size ≥ 2; otherwise 0
df['is_flattened'] = df['group_key'].apply(lambda x: 1 if group_counts[x] >= 2 else 0)

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
    activities = sorted(group['Activity'].unique())
    merged_activity = ';'.join(activities)
    merged_activities.append(merged_activity)
    label = group['label'].iloc[0] if 'label' in group.columns else None
    merged_events = pd.DataFrame({
        'Case': [name[0]],
        'Timestamp': [name[1]],
        'Activity': [merged_activity],
        'label': [label]
    })
    merged_events['group_key'] = name[0] + '_' + name[1]
    merged_events['is_flattened'] = 1
    merged_events['Activity'] = merged_activity
    merged_events['label'] = label
    merged_events['Timestamp'] = pd.to_datetime(merged_events['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Create a new DataFrame with merged records
merged_events = pd.concat([merged_events])

# Concatenate `normal_events` with merged `flattened_events`
final_df = pd.concat([normal_events, merged_events])

# Sort final DataFrame by `Case` and `Timestamp`
final_df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Drop helper columns (`group_key`, `is_flattened`)
final_df.drop(columns=['group_key', 'is_flattened'], inplace=True)

# If `label` column exists:
if 'label' in final_df.columns:
    # Define `y_true`: 1 if `label` not null/empty, else 0.
    y_true = final_df['label'].notnull().astype(int)
    # Define `y_pred`: value of `is_flattened`.
    y_pred = final_df['is_flattened']
    # Compute precision, recall, and F1-score using sklearn metrics.
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
    if precision >= 0.6:
        print("✓")
    else:
        print("✗")
else:
    print("No labels available for metric calculation.")

# Count and print:
# * Total flattened groups detected.
# * Total events marked as flattened.
# * Percentage of flattened events.
total_flattened_groups = len(final_df[final_df['is_flattened'] == 1])
total_events = len(final_df)
percentage_flattened = (total_flattened_groups / total_events) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Replace original flattened events with merged results.
# Keep all normal events unchanged.
final_df = final_df.copy()

# Select columns: `Case`, `Activity`, `Timestamp`, and `label` (if present).
final_df = final_df[['Case', 'Activity', 'Timestamp', 'label']]

# Ensure timestamp format `YYYY-MM-DD HH:MM:SS`.
final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')

# Save output CSV as:
output_path = 'data/pub/pub_form_based_cleaned_run1.csv'
final_df.to_csv(output_path, index=False)

# Print summary of what was processed for run 1
print(f"Run 1: Processed dataset saved to: {output_path}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: form_based")

# Print the path where the data was saved
print(f"Data saved to: {output_path}")

# Print the shape of the final dataset
print(f"Final dataset shape: {final_df.shape}")