# Generated script for Pub-Collateral - Run 3
# Generated on: 2025-11-14T13:24:43.440103
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
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
collateral_suffix = ':collateral'

# Input and output file paths
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run3.csv'

# Required columns
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
optional_columns = ['Variant', 'Resource']

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit(1)

# Normalize column names
df.rename(columns={case_column: 'Case', activity_column: 'Activity', timestamp_column: 'Timestamp'}, inplace=True)

# Convert Timestamp to datetime
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
except Exception as e:
    print(f"Error parsing timestamps: {e}")
    exit(1)

# Sort by Case and Timestamp
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify collateral activities
df['iscollateral'] = df['Activity'].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[_\-\s]+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case_id, case_group in df.groupby('Case'):
    case_indices = case_group.index.tolist()
    i = 0
    while i < len(case_indices):
        cluster = []
        cluster_start_time = case_group.loc[case_indices[i], 'Timestamp']
        base_activity = case_group.loc[case_indices[i], 'ProcessedActivity']
        mismatch_count = 0

        for j in range(i, len(case_indices)):
            event_time = case_group.loc[case_indices[j], 'Timestamp']
            time_diff = (event_time - cluster_start_time).total_seconds()

            if time_diff > time_threshold:
                break

            current_activity = case_group.loc[case_indices[j], 'ProcessedActivity']
            if current_activity == base_activity:
                cluster.append(case_indices[j])
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster.append(case_indices[j])
                else:
                    break

        # Filter cluster to dominant base activity
        if len(cluster) >= min_matching_events:
            cluster_activities = df.loc[cluster, 'ProcessedActivity']
            dominant_activity = cluster_activities.value_counts().idxmax()
            valid_cluster = [idx for idx in cluster if df.loc[idx, 'ProcessedActivity'] == dominant_activity]

            # Mark events for removal
            if valid_cluster:
                unsuffixed_event = next((idx for idx in valid_cluster if df.loc[idx, 'Activity'] == df.loc[idx, 'BaseActivity']), None)
                if unsuffixed_event is not None:
                    df.loc[unsuffixed_event, 'is_collateral_event'] = 0
                else:
                    df.loc[valid_cluster[0], 'is_collateral_event'] = 0

                for idx in valid_cluster:
                    if idx != unsuffixed_event:
                        df.loc[idx, 'is_collateral_event'] = 1

                df.loc[valid_cluster, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1

        i += len(cluster)

# Detection metrics
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
    print("No labels available for metric calculation. Metrics set to 0.0000.")

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_removed = df['is_collateral_event'].sum()
total_clean = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_removed}")
print(f"Events to be removed: {total_removed}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
df_fixed['Activity_fixed'] = df_fixed['Activity']
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
output_columns = ['Case', 'Timestamp', 'Variant', 'Activity', 'Activity_fixed', label_column] if label_column in df.columns else ['Case', 'Timestamp', 'Variant', 'Activity', 'Activity_fixed']
output_columns = [col for col in output_columns if col in df_fixed.columns]
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Total events (original): {df.shape[0]}")
print(f"Run 3: Total events (after removal): {df_fixed.shape[0]}")
print(f"Run 3: Events removed: {df.shape[0] - df_fixed.shape[0]} ({((df.shape[0] - df_fixed.shape[0]) / df.shape[0]) * 100:.2f}%)")
print(f"Run 3: Unique activities before removal: {df['Activity'].nunique()}")
print(f"Run 3: Unique activities after removal: {df_fixed['Activity'].nunique()}")