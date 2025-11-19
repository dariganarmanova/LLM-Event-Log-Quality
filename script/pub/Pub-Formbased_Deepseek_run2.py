# Generated script for Pub-Formbased - Run 2
# Generated on: 2025-11-14T13:27:24.781214
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
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
df = pd.read_csv('data/pub/Pub-Formbased.csv')
print(f"Run 2: Original dataset shape: {df.shape}")

# Standardize column names
df = df.rename(columns={
    'CaseID': 'Case',
    'case': 'Case',
    'event': 'Activity',
    'time': 'Timestamp',
    'timestamp': 'Timestamp'
})

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values(by=['Case', 'Timestamp'])

# Create group_key
df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp'].astype(str)

# Identify flattened events
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(lambda x: 1 if group_sizes[x] >= 2 else 0)

# Split into normal and flattened events
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Merge flattened activities
def merge_activities(group):
    activities = sorted(group['Activity'].unique())
    merged_activity = ';'.join(activities)
    first_row = group.iloc[0].copy()
    first_row['Activity'] = merged_activity
    if 'label' in group.columns:
        first_row['label'] = group['label'].iloc[0]
    return first_row

merged_flattened = flattened_events.groupby(['Case', 'Timestamp']).apply(merge_activities).reset_index(drop=True)

# Combine normal and merged flattened events
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)

# Sort final DataFrame
final_df = final_df.sort_values(by=['Case', 'Timestamp'])

# Drop helper columns
final_df = final_df.drop(columns=['group_key', 'is_flattened'])

# Calculate metrics if label column exists
if 'label' in final_df.columns:
    y_true = (~final_df['label'].isna() & (final_df['label'] != '')).astype(int)
    y_pred = df['is_flattened'].values[:len(y_true)]
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
total_flattened_groups = len(flattened_events.groupby(['Case', 'Timestamp']))
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Summary statistics
total_events = len(final_df)
unique_activities_before = df['Activity'].nunique()
unique_activities_after = final_df['Activity'].nunique()
reduction_percentage = ((len(df) - len(final_df)) / len(df)) * 100
print(f"Total number of events: {total_events}")
print(f"Number of flattened (merged) events detected: {total_flattened_groups}")
print(f"Number of unique activities before merging: {unique_activities_before}")
print(f"Number of unique activities after merging: {unique_activities_after}")
print(f"Total reduction percentage: {reduction_percentage:.2f}%")

# Print sample of merged activities
merged_samples = final_df[final_df['Activity'].str.contains(';')].head(10)
print("Sample of merged activities:")
for _, row in merged_samples.iterrows():
    print(f"Case: {row['Case']}, Timestamp: {row['Timestamp']}, Activity: {row['Activity']}")

# Save the processed data
output_path = 'data/pub/pub_form_based_cleaned_run2.csv'
final_df.to_csv(output_path, index=False)

# Print summary
print(f"Run 2: Processed dataset saved to: {output_path}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: form_based")