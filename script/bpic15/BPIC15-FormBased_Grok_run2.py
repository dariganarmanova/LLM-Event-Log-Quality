# Generated script for BPIC15-FormBased - Run 2
# Generated on: 2025-11-18T21:42:14.409332
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-FormBased.csv'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run2.csv'
dataset_name = 'bpic15'
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

# Rename common variants for Case if necessary
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': case_column})

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Flattened Events
flattened_mask = df.groupby([case_column, timestamp_column])[case_column].transform('size') >= min_matching_events
df['is_flattened'] = flattened_mask.astype(int)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns and not df[label_column].isna().all():
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_mark = "✓" if precision >= 0.6 else "✗"
    print(f"{prec_mark} Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 3: Split dataset
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 7: Integrity Check
if len(flattened_events) > 0:
    total_flattened_groups = flattened_events.groupby([case_column, timestamp_column]).size().count()
else:
    total_flattened_groups = 0
total_flattened_events_count = len(flattened_events)
percentage_flattened = (total_flattened_events_count / len(df) * 100) if len(df) > 0 else 0
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events_count}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Step 4: Merge Flattened Activities
merged_df = pd.DataFrame()
if len(flattened_events) > 0:
    agg_dict = {
        activity_column: lambda x: ';'.join(sorted(x.unique()))
    }
    other_columns = [col for col in df.columns if col not in [case_column, timestamp_column, 'is_flattened']]
    for col in other_columns:
        if col != activity_column:
            agg_dict[col] = 'first'
    merged_df = flattened_events.groupby([case_column, timestamp_column]).agg(agg_dict).reset_index()

# Step 5: Combine and Sort
normal_clean = normal_events.drop('is_flattened', axis=1)
if len(merged_df) > 0:
    final_df = pd.concat([normal_clean, merged_df], ignore_index=True)
else:
    final_df = normal_clean
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Standardize Timestamp format to YYYY-MM-DD HH:MM:SS
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 10: Summary Statistics
total_events_after = len(final_df)
unique_acts_before = df[activity_column].nunique()
unique_acts_after = final_df[activity_column].nunique()
reduction_pct = ((len(df) - total_events_after) / len(df) * 100) if len(df) > 0 else 0
print(f"Total number of events: {total_events_after}")
print(f"Number of flattened (merged) events detected: {total_flattened_groups}")
print(f"Number of unique activities before vs after merging: {unique_acts_before} vs {unique_acts_after}")
print(f"Total reduction percentage: {reduction_pct:.2f}%")
print(f"Output file path: {output_file}")
if len(merged_df) > 0:
    print("Sample of up to 10 merged activities (`;` separated):")
    for i, act in enumerate(merged_df[activity_column].head(10)):
        print(f"  - {act}")
else:
    print("No merged activities found.")

# Step 9: Save Output
final_df.to_csv(output_file, index=False)

# Required summary prints
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: form_based")