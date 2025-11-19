# Generated script for BPIC15-FormBased - Run 2
# Generated on: 2025-11-13T14:32:10.199427
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
input_file = 'data/bpic15/BPIC15-FormBased.csv'
df = pd.read_csv(input_file)

# Ensure all required columns exist; rename common variants
df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')
df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Create group_key: combination of Case and Timestamp values
df['group_key'] = df['Case'] + '_' + df['Timestamp']

# Count occurrences per group using groupby
group_counts = df.groupby('group_key')['Case'].count()

# Mark is_flattened = 1 if group size ≥ 2; otherwise 0
df['is_flattened'] = df['group_key'].map(group_counts) >= 2

# Split dataset: normal_events and flattened_events
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
flattened_events['Activity'] = flattened_events.groupby('group_key')['Activity'].transform(lambda x: ';'.join(sorted(x)))

# Keep the first label (if present) for each group
flattened_events['label'] = flattened_events.groupby('group_key')['label'].transform(lambda x: x.iloc[0])

# Create a new DataFrame with merged records
merged_events = pd.concat([normal_events, flattened_events[['Case', 'Timestamp', 'Activity', 'label']]])

# Concatenate normal_events with merged flattened_events
final_df = pd.concat([normal_events, merged_events])

# Sort final DataFrame by Case and Timestamp
final_df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Drop helper columns (group_key, is_flattened)
final_df.drop(columns=['group_key', 'is_flattened'], inplace=True)

# Calculate detection metrics (if label column exists)
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
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
    print(f"{'✓' if precision >= 0.6 else '✗'}")
else:
    print("No labels available for metric calculation.")

# Integrity check
flattened_groups = final_df[final_df['is_flattened'] == 1].shape[0]
total_flattened_events = final_df['is_flattened'].sum()
percentage_flattened = (total_flattened_events / final_df.shape[0]) * 100
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Save output
output_path = 'data/bpic15/bpic15_form_based_cleaned_run2.csv'
final_df[['Case', 'Activity', 'Timestamp', 'label']].to_csv(output_path, index=False)

# Print summary
print(f"Run 2: Processed dataset saved to: {output_path}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: form_based")