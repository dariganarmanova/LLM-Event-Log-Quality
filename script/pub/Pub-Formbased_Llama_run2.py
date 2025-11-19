# Generated script for Pub-Formbased - Run 2
# Generated on: 2025-11-14T13:29:20.819763
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
input_file = 'data/pub/Pub-Formbased.csv'
df = pd.read_csv(input_file)

# Rename common variants
df = df.rename(columns={'CaseID': 'Case'})

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df = df.sort_values(by=['Case', 'Timestamp'])

# Create group_key: combination of Case and Timestamp values
df['group_key'] = df['Case'] + '_' + df['Timestamp'].dt.strftime('%H:%M:%S')

# Count occurrences per group using groupby
group_counts = df.groupby('group_key')['Case'].count()

# Mark is_flattened = 1 if group size ≥ 2; otherwise 0
df['is_flattened'] = np.where(group_counts >= min_matching_events, 1, 0)

# Split dataset: normal_events and flattened_events
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Group flattened events by (Case, Timestamp)
flattened_groups = flattened_events.groupby(['Case', 'Timestamp'])

# For each group: merge all Activity values alphabetically using ; as separator
merged_activities = []
for name, group in flattened_groups:
    activity_list = sorted(group['Activity'].unique().tolist())
    merged_activity = ';'.join(activity_list)
    merged_activities.append(merged_activity)

# Create a new DataFrame with merged records
merged_flattened_events = pd.DataFrame({
    'Case': flattened_events['Case'].unique(),
    'Timestamp': flattened_events['Timestamp'].unique(),
    'Activity': merged_activities,
    'label': flattened_events['label'].unique()
})

# Concatenate normal_events with merged flattened_events
final_df = pd.concat([normal_events, merged_flattened_events])

# Sort final DataFrame by Case and Timestamp
final_df = final_df.sort_values(by=['Case', 'Timestamp'])

# Drop helper columns (group_key, is_flattened)
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# If label column exists: calculate precision, recall, and F1-score
if 'label' in final_df.columns:
    y_true = final_df['label'].notnull().astype(int)
    y_pred = final_df['is_flattened']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("No labels available for metric calculation.")

# Count and print: total flattened groups detected, total events marked as flattened, percentage of flattened events
total_flattened_groups = len(flattened_groups.groups)
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Save output CSV
output_path = 'data/pub/pub_form_based_cleaned_run2.csv'
final_df.to_csv(output_path, index=False)

# Print summary statistics
print(f"Total number of events: {len(df)}")
print(f"Number of flattened (merged) events detected: {total_flattened_events}")
print(f"Number of unique activities before vs after merging: {len(df['Activity'].unique())} vs {len(final_df['Activity'].unique())}")
print(f"Total reduction percentage: {(1 - (total_flattened_events / len(df))) * 100:.2f}%")
print(f"Output file path: {output_path}")

# Print sample of up to 10 merged activities (; separated)
merged_activities_sample = ';'.join(final_df['Activity'].unique()[:10])
print(f"Merged activities sample: {merged_activities_sample}")