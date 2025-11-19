# Generated script for BPIC11-FormBased - Run 3
# Generated on: 2025-11-13T11:20:36.178322
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the dataset.")

# Convert timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by case and timestamp
df = df.sort_values(by=[case_column, timestamp_column])

# Identify flattened events
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_sizes).apply(lambda x: 1 if x >= 2 else 0)

# Split into normal and flattened events
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
def merge_activities(group):
    activities = sorted(group[activity_column].unique())
    merged_activity = ';'.join(activities)
    first_row = group.iloc[0].copy()
    first_row[activity_column] = merged_activity
    if label_column in group.columns:
        first_row[label_column] = group[label_column].iloc[0]
    return first_row

merged_flattened = flattened_events.groupby('group_key').apply(merge_activities).reset_index(drop=True)

# Combine normal and merged flattened events
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column])
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# Calculate metrics if label column exists
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No labels available for metric calculation.")

# Integrity check
total_flattened_groups = len(flattened_events['group_key'].unique())
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Summary statistics
total_events_before = len(df)
total_events_after = len(final_df)
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
reduction_percentage = ((total_events_before - total_events_after) / total_events_before) * 100

print(f"Run 3: Total number of events before merging: {total_events_before}")
print(f"Run 3: Number of flattened (merged) events detected: {total_flattened_events}")
print(f"Run 3: Number of unique activities before merging: {unique_activities_before}")
print(f"Run 3: Number of unique activities after merging: {unique_activities_after}")
print(f"Run 3: Total reduction percentage: {reduction_percentage:.2f}%")

# Display sample of merged activities
sample_merged = final_df[final_df[activity_column].str.contains(';')].head(10)
if not sample_merged.empty:
    print("Sample of merged activities:")
    for _, row in sample_merged.iterrows():
        print(f"Case: {row[case_column]}, Timestamp: {row[timestamp_column]}, Activity: {row[activity_column]}")

# Save the final processed data
final_df.to_csv(output_file, index=False)

# Print final summary
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: form_based")