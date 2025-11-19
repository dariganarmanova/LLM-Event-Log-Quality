# Generated script for BPIC15-CollateralEvent - Run 1
# Generated on: 2025-11-13T14:41:44.594804
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
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run1.csv'

# Column names
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    required_columns = {case_column, activity_column, timestamp_column}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

    # Step 2: Identify Collateral Activities
    df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
    df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '', regex=False)

    # Step 3: Preprocess Activity Names
    def preprocess_activity(activity):
        activity = activity.lower() if not case_sensitive else activity
        activity = re.sub(activity_suffix_pattern, '', activity)
        activity = re.sub(r'[_\-\s]+', ' ', activity).strip()
        return activity

    df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

    # Step 4: Initialize Tracking Columns
    df['CollateralGroup'] = -1
    df['is_collateral_event'] = 0
    cluster_counter = 0

    # Step 5: Sliding Window Clustering
    for case_id, group in df.groupby(case_column):
        group_indices = group.index.tolist()
        i = 0
        while i < len(group_indices):
            cluster = [group_indices[i]]
            cluster_start_time = group.loc[group_indices[i], timestamp_column]
            base_activity = group.loc[group_indices[i], 'ProcessedActivity']
            mismatch_count = 0

            for j in range(i + 1, len(group_indices)):
                time_diff = (group.loc[group_indices[j], timestamp_column] - cluster_start_time).total_seconds()
                if time_diff > time_threshold:
                    break

                current_activity = group.loc[group_indices[j], 'ProcessedActivity']
                if current_activity == base_activity:
                    cluster.append(group_indices[j])
                else:
                    mismatch_count += 1
                    if mismatch_count <= max_mismatches:
                        cluster.append(group_indices[j])
                    else:
                        break

            # Filter cluster to dominant base activity
            cluster_activities = group.loc[cluster, 'ProcessedActivity']
            dominant_base = cluster_activities.mode().iloc[0]
            cluster = [idx for idx in cluster if group.loc[idx, 'ProcessedActivity'] == dominant_base]

            # Validate cluster
            if len(cluster) >= min_matching_events:
                unsuffixed_events = [idx for idx in cluster if group.loc[idx, 'BaseActivity'] == dominant_base]
                if unsuffixed_events:
                    keep_idx = unsuffixed_events[0]
                else:
                    keep_idx = cluster[0]

                for idx in cluster:
                    if idx != keep_idx:
                        df.at[idx, 'is_collateral_event'] = 1
                df.loc[cluster, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1

            i += len(cluster)

    # Step 6: Calculate Detection Metrics
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
        print("=== Detection Performance Metrics ===")
        print("No labels available for metric calculation")
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")

    # Step 7: Integrity Check
    total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
    total_marked = df['is_collateral_event'].sum()
    total_clean = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
    print(f"Total collateral clusters detected: {total_clusters}")
    print(f"Total events marked as collateral: {total_marked}")
    print(f"Events to be removed: {total_marked}")
    print(f"Clean events untouched: {total_clean}")

    # Step 8: Remove Collateral Events
    df_fixed = df[df['is_collateral_event'] == 0].copy()

    # Step 9: Save Output
    df_fixed['Activity_fixed'] = df_fixed[activity_column]
    df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_fixed.to_csv(output_file, index=False)

    # Step 10: Summary Statistics
    print(f"Total events (original): {len(df)}")
    print(f"Total events (after removal): {len(df_fixed)}")
    print(f"Events removed: {len(df) - len(df_fixed)} ({((len(df) - len(df_fixed)) / len(df)) * 100:.2f}%)")
    print(f"Unique activities before removal: {df[activity_column].nunique()}")
    print(f"Unique activities after removal: {df_fixed[activity_column].nunique()}")
    print(f"Run 1: Processed dataset saved to: {output_file}")
    print(f"Run 1: Final dataset shape: {df_fixed.shape}")

except Exception as e:
    print(f"An error occurred: {e}")