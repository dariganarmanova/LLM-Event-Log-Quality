# Generated script for Credit-Formbased - Run 1
# Generated on: 2025-11-13T16:19:21.647663
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

# Rename common variants
df = df.rename(columns={'CaseID': 'Case'})

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df = df.sort_values(by=['Case', 'Timestamp'])

# Identify flattened events
group_key = df['Case'] + '_' + df['Timestamp'].dt.strftime('%H:%M:%S')
df['group_key'] = group_key
df['is_flattened'] = df.groupby('group_key')['Activity'].transform('count').ge(min_matching_events)

# Preprocess flattened groups
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
flattened_events = flattened_events.groupby(['Case', 'Timestamp'])['Activity'].apply(lambda x: ';'.join(sorted(x.unique()))).reset_index()

# Combine and sort
df = pd.concat([normal_events, flattened_events]).sort_values(by=['Case', 'Timestamp']).drop(columns=['group_key', 'is_flattened'])

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

# Integrity check
total_flattened_groups = df[df['is_flattened'] == 1].groupby('group_key').size().sum()
total_events_marked_as_flattened = df[df['is_flattened'] == 1].shape[0]
percentage_of_flattened_events = (total_events_marked_as_flattened / df.shape[0]) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_events_marked_as_flattened}")
print(f"Percentage of flattened events: {percentage_of_flattened_events:.2f}%")

# Fix events
df = df.drop(columns=['is_flattened'])

# Save output
output_path = 'data/credit/credit_form_based_cleaned_run1.csv'
df.to_csv(output_path, index=False)

# Summary statistics
print(f"Total number of events: {df.shape[0]}")
print(f"Number of flattened (merged) events detected: {total_events_marked_as_flattened}")
print(f"Number of unique activities before vs after merging: {len(df['Activity'].unique())} vs {len(set(df['Activity'].str.split(';').explode().values))}")
print(f"Total reduction percentage: {(1 - (total_events_marked_as_flattened / df.shape[0])) * 100:.2f}%")
print(f"Output file path: {output_path}")

# Print a sample of up to 10 merged activities
merged_activities = df['Activity'].str.split(';').explode().value_counts().head(10)
print(f"Sample of merged activities: {merged_activities}")