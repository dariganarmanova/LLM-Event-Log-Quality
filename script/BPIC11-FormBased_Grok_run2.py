# Generated script for BPIC11-FormBased - Run 2
# Generated on: 2025-11-18T22:27:22.649555
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run2.csv'
dataset_name = 'bpic11'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
min_matching_events = 2

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist (assume standard names for this dataset)
# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Create group_key for exact timestamp matching
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Count occurrences per group
df['count_per_group'] = df.groupby('group_key')['group_key'].transform('size')

# Mark flattened events
df['is_flattened'] = (df['count_per_group'] >= min_matching_events).astype(int)

# #6. Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column] != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    status = "✓" if prec >= 0.6 else "✗"
    print(f"{status} Precision threshold (>= 0.6) { 'met' if status == '✓' else 'not met' }")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("✗ Precision threshold (>= 0.6) not met")
    print("No labels available for metric calculation.")

# #7. Integrity Check
flattened_groups = df[df['is_flattened'] == 1]['group_key'].nunique()
total_flattened_events = df['is_flattened'].sum()
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# #3 & #4. Preprocess and Merge Flattened Activities
normal_events = df[df['is_flattened'] == 0].copy().drop(columns=['group_key', 'count_per_group', 'is_flattened'])

flattened_mask = df['is_flattened'] == 1
if flattened_mask.any():
    flattened_df = df[flattened_mask]
    agg_dict = {activity_column: lambda x: ';'.join(sorted(x))}
    for col in df.columns:
        if col not in [case_column, timestamp_column, activity_column, 'group_key', 'count_per_group', 'is_flattened']:
            agg_dict[col] = 'first'
    merged_df = flattened_df.groupby([case_column, timestamp_column])[list(agg_dict.keys())].agg(agg_dict).reset_index()
else:
    # Empty merged if no flattened
    merged_df = pd.DataFrame(columns=normal_events.columns if len(normal_events) > 0 else [case_column, activity_column, timestamp_column])

# #5. Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# #10. Summary Statistics
original_events = len(df)
final_events = len(final_df)
reduction_pct = ((original_events - final_events) / original_events * 100) if original_events > 0 else 0
unique_before = df[activity_column].nunique()
unique_after = final_df[activity_column].nunique()
print(f"Total number of events: {final_events}")
print(f"Number of flattened (merged) events detected: {flattened_groups}")
print(f"Number of unique activities before vs after merging: {unique_before} vs {unique_after}")
print(f"Total reduction percentage: {reduction_pct:.2f}%")

# Sample of up to 10 merged activities
merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10).tolist()
print("Sample of up to 10 merged activities (`;` separated):")
for act in merged_samples:
    print(f"  - {act}")

# Format timestamp for output
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# #9. Save Output
final_df.to_csv(output_file, index=False)

# Required prints
print(f"Run 2: Processed dataset saved to: data/bpic11/bpic11_form_based_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: form_based")