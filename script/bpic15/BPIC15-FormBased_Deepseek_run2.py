# Generated script for BPIC15-FormBased - Run 2
# Generated on: 2025-11-13T14:30:40.351629
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-FormBased.csv'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run2.csv'
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
print(f"Run 2: Original dataset shape: {df.shape}")

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

# Sort by case and timestamp
df = df.sort_values(by=[case_column, timestamp_column])

# Create group key
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)

# Mark flattened events
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_sizes).ge(2).astype(int)

# Split into normal and flattened events
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
def merge_activities(group):
    merged_activity = ';'.join(sorted(group[activity_column].unique()))
    first_row = group.iloc[0].copy()
    first_row[activity_column] = merged_activity
    if label_column in group.columns:
        first_row[label_column] = group[label_column].iloc[0]
    return first_row

merged_flattened = flattened_events.groupby(['group_key']).apply(merge_activities).reset_index(drop=True)

# Combine normal and merged events
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column])

# Drop helper columns
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# Calculate metrics if label column exists
if label_column in df.columns:
    y_true = (~df[label_column].isna()).astype(int)
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

# Integrity check
total_flattened_groups = len(flattened_events['group_key'].unique())
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Summary statistics
total_events_original = len(df)
total_events_final = len(final_df)
unique_activities_original = df[activity_column].nunique()
unique_activities_final = final_df[activity_column].nunique()
reduction_percentage = ((total_events_original - total_events_final) / total_events_original) * 100

print(f"Run 2: Original number of events: {total_events_original}")
print(f"Run 2: Final number of events: {total_events_final}")
print(f"Run 2: Unique activities before merging: {unique_activities_original}")
print(f"Run 2: Unique activities after merging: {unique_activities_final}")
print(f"Run 2: Total reduction percentage: {reduction_percentage:.2f}%")

# Save the final dataset
final_df.to_csv(output_file, index=False)

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: form_based")