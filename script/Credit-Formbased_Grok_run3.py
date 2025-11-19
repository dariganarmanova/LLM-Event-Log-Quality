# Generated script for Credit-Formbased - Run 3
# Generated on: 2025-11-18T19:23:56.783440
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Formbased.csv'
dataset_name = 'credit'
output_file = 'data/credit/credit_form_based_cleaned_run3.csv'
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
print(f"Run 3: Original dataset shape: {df.shape}")

# Handle column renaming for common variants
columns_lower = {col.lower(): col for col in df.columns}
if 'case' in columns_lower and columns_lower['case'] != 'Case':
    df.rename(columns={columns_lower['case']: 'Case'}, inplace=True)
if 'caseid' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)
if 'case id' in columns_lower:
    df.rename(columns={columns_lower['case id']: 'Case'}, inplace=True)
if 'activity' in columns_lower and columns_lower['activity'] != 'Activity':
    df.rename(columns={columns_lower['activity']: 'Activity'}, inplace=True)
if 'event' in columns_lower and 'activity' in columns_lower['event']:
    df.rename(columns={columns_lower['event']: 'Activity'}, inplace=True)
if 'timestamp' in columns_lower and columns_lower['timestamp'] != 'Timestamp':
    df.rename(columns={columns_lower['timestamp']: 'Timestamp'}, inplace=True)
if 'time' in columns_lower:
    df.rename(columns={columns_lower['time']: 'Timestamp'}, inplace=True)
if 'event_time' in columns_lower:
    df.rename(columns={columns_lower['event_time']: 'Timestamp'}, inplace=True)

# Ensure required columns exist
required_cols = ['Case', 'Activity', 'Timestamp']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Check for label column
label_exists = 'label' in df.columns

# Check for optional columns
variant_col = 'Variant' if 'Variant' in df.columns else None
resource_col = 'Resource' if 'Resource' in df.columns else None

# Define output columns
output_cols = ['Case', 'Activity', 'Timestamp']
if label_exists:
    output_cols.append('label')
if variant_col:
    output_cols.append(variant_col)
if resource_col:
    output_cols.append(resource_col)

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp
df.sort_values(['Case', 'Timestamp'], inplace=True)

# Step 2: Identify Flattened Events
df['group_size'] = df.groupby(['Case', 'Timestamp']).transform('size')
df['is_flattened'] = (df['group_size'] >= min_matching_events).astype(int)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_exists:
    df['y_true'] = ((df['label'].notna()) & (df['label'] != '')).astype(int)
    y_true = df['y_true'].values
    y_pred = df['is_flattened'].values
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (>= 0.6) met")
    else:
        print("✗ Precision threshold (>= 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("✗ No labels available for metric calculation.")

# Step 7: Integrity Check
flattened_mask = df['is_flattened'] == 1
total_flattened_events = flattened_mask.sum()
num_flattened_groups = df[flattened_mask].groupby(['Case', 'Timestamp']).size().count()
percentage_flattened = (total_flattened_events / len(df) * 100) if len(df) > 0 else 0
print(f"Total flattened groups detected: {num_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Step 3: Preprocess Flattened Groups
normal_events = df[~flattened_mask][output_cols].copy()
flattened_events = df[flattened_mask].copy()

# Step 4: Merge Flattened Activities
def merge_group(group):
    activities = sorted(group['Activity'].dropna().astype(str).tolist())
    merged_activity = ';'.join(activities)
    row = {
        'Case': group['Case'].iloc[0],
        'Timestamp': group['Timestamp'].iloc[0],
        'Activity': merged_activity
    }
    if label_exists:
        row['label'] = group['label'].iloc[0]
    if variant_col:
        row[variant_col] = group[variant_col].iloc[0]
    if resource_col:
        row[resource_col] = group[resource_col].iloc[0]
    return pd.Series(row)

if len(flattened_events) > 0:
    merged_df = flattened_events.groupby(['Case', 'Timestamp']).apply(merge_group).reset_index(drop=True)
else:
    merged_df = pd.DataFrame(columns=output_cols)

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
if not final_df.empty:
    final_df.sort_values(['Case', 'Timestamp'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

# Step 8: Fix Events (already done via merging)

# Format Timestamp for output
if not final_df.empty:
    final_df['Timestamp'] = final_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 9: Save Output
final_df.to_csv(output_file, index=False)

# Step 10: Summary Statistics
total_events_after = len(final_df)
num_flattened_merged = len(merged_df)
unique_acts_before = df['Activity'].nunique()
unique_acts_after = final_df['Activity'].nunique()
reduction_pct = ((len(df) - total_events_after) / len(df) * 100) if len(df) > 0 else 0
print(f"Total number of events: {total_events_after}")
print(f"Number of flattened (merged) events detected: {num_flattened_merged}")
print(f"Number of unique activities before vs after merging: {unique_acts_before} vs {unique_acts_after}")
print(f"Total reduction percentage: {reduction_pct:.2f}%")
print(f"Output file path: {output_file}")

# Sample of up to 10 merged activities
merged_samples = final_df[final_df['Activity'].str.contains(';', na=False)]['Activity'].head(10).tolist()
print("Sample of up to 10 merged activities (;-separated):")
if merged_samples:
    for act in merged_samples:
        print(f"  - {act}")
else:
    print("  - No merged activities found.")

# Required prints
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: {dataset_name}")
print(f"Run 3: Task type: form_based")