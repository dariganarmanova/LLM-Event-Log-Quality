# Generated script for BPIC15-CollateralEvent - Run 2
# Generated on: 2025-11-13T14:42:07.208187
# Model: gpt-4o-2024-11-20

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

# Input and output file paths
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run2.csv'

# Column names
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

# Load the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    print(f"Error: Missing required columns. Expected columns: {required_columns}")
    exit(1)

# Convert timestamp to datetime
try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
except Exception as e:
    print(f"Error converting timestamps: {e}")
    exit(1)

# Drop rows with invalid timestamps
df = df.dropna(subset=[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify collateral activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '', regex=False)

# Preprocess activity names
def preprocess_activity(activity):
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[_\-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case_id, case_group in df.groupby(case_column):
    case_group = case_group.reset_index()
    n = len(case_group)
    i = 0
    while i < n:
        cluster = [i]
        cluster_start_time = case_group.loc[i, timestamp_column]
        base_activity = case_group.loc[i, 'ProcessedActivity']
        mismatch_count = 0

        # Expand cluster
        for j in range(i + 1, n):
            time_diff = (case_group.loc[j, timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            if case_group.loc[j, 'ProcessedActivity'] == base_activity:
                cluster.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break

        # Filter cluster to dominant base activity
        cluster_activities = case_group.loc[cluster, 'ProcessedActivity']
        dominant_activity = cluster_activities.mode()[0]
        valid_cluster = case_group.loc[cluster][case_group['ProcessedActivity'] == dominant_activity]

        # Validate cluster
        if len(valid_cluster) >= min_matching_events:
            valid_indices = valid_cluster.index
            unsuffixed_event = valid_cluster[valid_cluster['BaseActivity'] == dominant_activity]
            if not unsuffixed_event.empty:
                keep_index = unsuffixed_event.index[0]
            else:
                keep_index = valid_indices[0]

            df.loc[valid_indices, 'is_collateral_event'] = 1
            df.loc[keep_index, 'is_collateral_event'] = 0
            df.loc[valid_indices, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

        i = cluster[-1] + 1

# Calculate detection metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].fillna('').str.contains('collateral', case=False).astype(int)
    y_true = df['y_true']
    y_pred = df['is_collateral_event']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation")

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_marked = df['is_collateral_event'].sum()
total_clean = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_marked}")
print(f"Clean events not modified: {total_clean}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Run 2: Total events (original): {len(df)}")
print(f"Run 2: Total events (after removal): {len(df_fixed)}")
print(f"Run 2: Events removed: {len(df) - len(df_fixed)} ({((len(df) - len(df_fixed)) / len(df)) * 100:.2f}%)")
print(f"Run 2: Unique activities before removal: {df[activity_column].nunique()}")
print(f"Run 2: Unique activities after removal: {df_fixed[activity_column].nunique()}")
print(f"Run 2: Processed dataset saved to: {output_file}")