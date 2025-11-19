# Generated script for BPIC11-FormBased - Run 1
# Generated on: 2025-11-13T11:19:10.400785
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Standardize column names
df = df.rename(columns={
    'CaseID': case_column,
    'case:concept:name': case_column,
    'concept:name': activity_column,
    'time:timestamp': timestamp_column,
    'Label': label_column
})

# Convert timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column])

# Step 2: Identify Flattened Events
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_sizes).apply(lambda x: 1 if x >= min_matching_events else 0)

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Step 4: Merge Flattened Activities
def merge_activities(group):
    activities = sorted(group[activity_column].unique())
    merged_activity = ';'.join(activities)
    first_label = group[label_column].iloc[0] if label_column in group.columns else None
    return pd.Series({
        case_column: group[case_column].iloc[0],
        timestamp_column: group[timestamp_column].iloc[0],
        activity_column: merged_activity,
        **({label_column: first_label} if label_column in group.columns else {})
    })

merged_flattened = flattened_events.groupby('group_key').apply(merge_activities).reset_index(drop=True)

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column])
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# Step 6: Calculate Detection Metrics
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No labels available for metric calculation.")

# Step 7: Integrity Check
total_flattened_groups = len(flattened_events['group_key'].unique())
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Step 8: Fix Events (already done in Step 5)

# Step 9: Save Output
final_df.to_csv(output_file, index=False)

# Step 10: Summary Statistics
total_events_original = len(df)
total_events_final = len(final_df)
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
reduction_percentage = ((total_events_original - total_events_final) / total_events_original) * 100

print(f"Run 1: Total number of events before merging: {total_events_original}")
print(f"Run 1: Total number of events after merging: {total_events_final}")
print(f"Run 1: Number of unique activities before merging: {unique_activities_before}")
print(f"Run 1: Number of unique activities after merging: {unique_activities_after}")
print(f"Run 1: Total reduction percentage: {reduction_percentage:.2f}%")
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: form_based")