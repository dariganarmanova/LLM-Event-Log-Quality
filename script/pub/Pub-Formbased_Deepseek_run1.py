# Generated script for Pub-Formbased - Run 1
# Generated on: 2025-11-14T13:26:48.269126
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/pub/Pub-Formbased.csv'
output_file = 'data/pub/pub_form_based_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
variant_column = 'Variant'
resource_column = 'Resource'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Standardize column names if necessary
df.columns = df.columns.str.replace('CaseID', case_column)
df.columns = df.columns.str.replace('ActivityName', activity_column)
df.columns = df.columns.str.replace('Timestamp', timestamp_column)

# Convert timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column])

# Step 2: Identify Flattened Events
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].astype(str)
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_sizes).apply(lambda x: 1 if x >= 2 else 0)

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Step 4: Merge Flattened Activities
merged_activities = flattened_events.groupby([case_column, timestamp_column, 'group_key']).agg({
    activity_column: lambda x: ';'.join(sorted(x)),
    label_column: 'first' if label_column in df.columns else None,
    variant_column: 'first' if variant_column in df.columns else None,
    resource_column: 'first' if resource_column in df.columns else None
}).reset_index()

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_activities], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column])
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# Step 6: Calculate Detection Metrics
if label_column in df.columns:
    y_true = df[label_column].notnull().astype(int)
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

# Step 7: Integrity Check
total_flattened_groups = len(flattened_events['group_key'].unique())
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Step 8: Fix Events (already done in previous steps)

# Step 9: Save Output
final_df.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print(f"Run 1: Total number of events before merging: {len(df)}")
print(f"Run 1: Total number of events after merging: {len(final_df)}")
print(f"Run 1: Number of flattened (merged) events detected: {total_flattened_groups}")
print(f"Run 1: Number of unique activities before merging: {df[activity_column].nunique()}")
print(f"Run 1: Number of unique activities after merging: {final_df[activity_column].nunique()}")
reduction_percentage = ((len(df) - len(final_df)) / len(df)) * 100
print(f"Run 1: Total reduction percentage: {reduction_percentage:.2f}%")
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: form_based")