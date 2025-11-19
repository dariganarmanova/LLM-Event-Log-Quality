# Generated script for BPIC11-FormBased - Run 3
# Generated on: 2025-11-18T22:27:56.869121
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# File paths
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run3.csv'
dataset_name = 'bpic11'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Handle column renaming for common variants
column_mapping = {
    'Case ID': 'Case',
    'case_id': 'Case',
    'CaseID': 'Case',
    'caseid': 'Case',
    'Activity': 'Activity',
    'activity': 'Activity',
    'event': 'Activity',
    'Timestamp': 'Timestamp',
    'time:timestamp': 'Timestamp',
    'Complete Timestamp': 'Timestamp',
    'label': 'label'
}
df = df.rename(columns=column_mapping)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp
df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp'].astype(str)
group_counts = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(lambda x: 1 if group_counts.get(x, 0) >= min_matching_events else 0)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
has_label = 'label' in df.columns
if has_label:
    y_true = (df['label'].notna() & (df['label'].astype(str) != '')).astype(int)
    y_pred = df['is_flattened']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_check = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_check} Precision threshold (>= 0.6) {'met' if prec >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")
    print("✗ Precision threshold (>= 0.6) not met")

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 7: Integrity Check
flattened_groups = flattened_events['group_key'].nunique() if len(flattened_events) > 0 else 0
total_flattened_events = len(flattened_events)
total_events = len(df)
percentage = (total_flattened_events / total_events * 100) if total_events > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 4: Merge Flattened Activities
merged_rows = []
if len(flattened_events) > 0:
    for group, gdf in flattened_events.groupby('group_key'):
        activities = sorted([str(act).strip() for act in gdf['Activity'] if pd.notna(act) and str(act).strip() != ''])
        merged_activity = ';'.join(activities)
        first_row = gdf.iloc[0]
        merged_row = {
            'Case': first_row['Case'],
            'Timestamp': first_row['Timestamp'],
            'Activity': merged_activity
        }
        if 'label' in df.columns:
            merged_row['label'] = first_row['label']
        # Preserve other columns (e.g., Resource, Variant) from first row
        other_cols = [col for col in df.columns if col not in ['Case', 'Activity', 'Timestamp', 'label', 'group_key', 'is_flattened']]
        for col in other_cols:
            merged_row[col] = first_row[col]
        merged_rows.append(merged_row)
    merged_df = pd.DataFrame(merged_rows)
else:
    merged_df = pd.DataFrame(columns=df.columns)

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
cols_to_drop = ['group_key', 'is_flattened']
for col in cols_to_drop:
    if col in final_df.columns:
        final_df = final_df.drop(columns=[col])
final_df = final_df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)
# Standardize Timestamp format to YYYY-MM-DD HH:MM:SS
final_df['Timestamp'] = final_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 10: Summary Statistics
print(f"Total number of events: {len(final_df)}")
num_merged = len(merged_df)
print(f"Number of flattened (merged) events detected: {num_merged}")
unique_acts_before = df['Activity'].nunique()
unique_acts_after = final_df['Activity'].nunique()
print(f"Number of unique activities before vs after merging: {unique_acts_before} vs {unique_acts_after}")
reduction = (1 - len(final_df) / len(df)) * 100 if len(df) > 0 else 0
print(f"Total reduction percentage: {reduction:.2f}%")
print(f"Output file path: {output_file}")
# Sample up to 10 merged activities
merged_samples = final_df[final_df['Activity'].str.contains(';', na=False)]['Activity'].head(10).tolist()
print("Sample of up to 10 merged activities (;-separated):")
for i, act in enumerate(merged_samples, 1):
    print(f"{i}. {act}")

# Step 9: Save Output
final_df.to_csv(output_file, index=False)

# Required summary prints
print(f"Run 3: Processed dataset saved to: data/bpic11/bpic11_form_based_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: form_based")