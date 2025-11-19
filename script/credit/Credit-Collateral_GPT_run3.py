# Generated script for Credit-Collateral - Run 3
# Generated on: 2025-11-13T15:09:39.201630
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0  # seconds
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# File paths
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run3.csv'

# Column names
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    print(f"Error: Missing required columns in the dataset. Required columns: {required_columns}")
    exit()

# Convert Timestamp to datetime
try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    df = df.dropna(subset=[timestamp_column])  # Drop rows with invalid timestamps
except Exception as e:
    print(f"Error parsing timestamps: {e}")
    exit()

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify collateral activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '', regex=False)

# Preprocess activity names
def preprocess_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[_\-\s]+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case_id, case_group in df.groupby(case_column):
    case_indices = case_group.index.tolist()
    i = 0
    while i < len(case_indices):
        cluster = [case_indices[i]]
        cluster_start_time = case_group.loc[case_indices[i], timestamp_column]
        base_activity = case_group.loc[case_indices[i], 'ProcessedActivity']
        mismatch_count = 0

        for j in range(i + 1, len(case_indices)):
            time_diff = (case_group.loc[case_indices[j], timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            current_activity = case_group.loc[case_indices[j], 'ProcessedActivity']
            if current_activity == base_activity:
                cluster.append(case_indices[j])
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break

        # Filter cluster to dominant base activity
        cluster_activities = df.loc[cluster, 'ProcessedActivity']
        dominant_activity = cluster_activities.mode()[0]
        valid_cluster = cluster_activities[cluster_activities == dominant_activity].index.tolist()

        # Validate cluster
        if len(valid_cluster) >= min_matching_events:
            unsuffixed_event = df.loc[valid_cluster, activity_column].apply(lambda x: x == preprocess_activity(x)).any()
            if unsuffixed_event:
                keep_index = df.loc[valid_cluster, activity_column].apply(lambda x: x == preprocess_activity(x)).idxmax()
            else:
                keep_index = valid_cluster[0]

            df.loc[valid_cluster, 'is_collateral_event'] = 1
            df.loc[keep_index, 'is_collateral_event'] = 0
            df.loc[valid_cluster, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

        i += len(cluster)

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
total_collateral_events = df['is_collateral_event'].sum()
total_clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_collateral_events}")
print(f"Events to be removed: {total_collateral_events}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
try:
    df_fixed.to_csv(output_file, index=False)
    print(f"Run 3: Processed dataset saved to: {output_file}")
    print(f"Run 3: Final dataset shape: {df_fixed.shape}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Summary statistics
print(f"Total events (original): {df.shape[0]}")
print(f"Total events (after removal): {df_fixed.shape[0]}")
print(f"Events removed: {df.shape[0] - df_fixed.shape[0]} ({((df.shape[0] - df_fixed.shape[0]) / df.shape[0]) * 100:.2f}%)")
print(f"Unique activities before removal: {df[activity_column].nunique()}")
print(f"Unique activities after removal: {df_fixed[activity_column].nunique()}")