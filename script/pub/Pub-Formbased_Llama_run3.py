# Generated script for Pub-Formbased - Run 3
# Generated on: 2025-11-14T13:29:22.614302
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

# Ensure all required columns exist; rename common variants
df = df.rename(columns={'CaseID': 'Case'})

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify flattened events
group_key = df.groupby(['Case', 'Timestamp']).size().reset_index(name='count')
group_key['is_flattened'] = group_key['count'] >= min_matching_events

# Merge flattened events
df = pd.merge(df, group_key, on=['Case', 'Timestamp'], how='left')
df['is_flattened'] = df['is_flattened'].astype(int)

# Split dataset into normal and flattened events
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Merge flattened activities
flattened_events['Activity'] = flattened_events.groupby(['Case', 'Timestamp'])['Activity'].transform(lambda x: ';'.join(sorted(x.unique())))

# Combine and sort
df = pd.concat([normal_events, flattened_events]).sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

# Drop helper columns
df = df.drop(columns=['count', 'is_flattened'])

# Calculate detection metrics
if 'label' in df.columns:
    y_true = df['label'].notnull().astype(int)
    y_pred = df['is_flattened']
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
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")

# Integrity check
total_flattened_groups = df[df['is_flattened'] == 1].shape[0]
total_flattened_events = df[df['is_flattened'] == 1].shape[0]
percentage_flattened_events = (total_flattened_events / df.shape[0]) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened_events:.2f}%")

# Fix events
df = df.drop(columns=['count', 'is_flattened'])

# Save output
output_path = 'data/pub/pub_form_based_cleaned_run3.csv'
df.to_csv(output_path, index=False)

# Summary statistics
total_events = df.shape[0]
unique_activities_before = df['Activity'].str.split(';').explode().nunique()
unique_activities_after = df['Activity'].nunique()
total_reduction_percentage = ((unique_activities_before - unique_activities_after) / unique_activities_before) * 100
print(f"Total number of events: {total_events}")
print(f"Number of flattened (merged) events detected: {total_flattened_events}")
print(f"Number of unique activities before merging: {unique_activities_before}")
print(f"Number of unique activities after merging: {unique_activities_after}")
print(f"Total reduction percentage: {total_reduction_percentage:.2f}%")
print(f"Output file path: {output_path}")

# Print sample of up to 10 merged activities
merged_activities_sample = df['Activity'].str.split(';').explode().head(10)
print(f"Merged activities sample: {merged_activities_sample.tolist()}")