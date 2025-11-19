# Generated script for BPIC11-CollateralEvent - Run 1
# Generated on: 2025-11-13T11:37:15.042925
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

# Input and output file paths
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run1.csv'

# Columns
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
        activity = re.sub(r'[_\-]', ' ', activity)
        activity = re.sub(r'\s+', ' ', activity).strip()
        return activity

    df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

    # Step 4: Initialize Tracking Columns
    df['CollateralGroup'] = -1
    df['is_collateral_event'] = 0
    cluster_counter = 0

    # Step 5: Sliding Window Clustering
    for case_id, case_group in df.groupby(case_column):
        case_indices = case_group.index.tolist()
        i = 0
        while i < len(case_indices):
            cluster_start_idx = case_indices[i]
            cluster_start_time = case_group.loc[cluster_start_idx, timestamp_column]
            base_activity = case_group.loc[cluster_start_idx, 'ProcessedActivity']
            cluster_indices = [cluster_start_idx]
            mismatch_count = 0

            for j in range(i + 1, len(case_indices)):
                current_idx = case_indices[j]
                time_diff = (case_group.loc[current_idx, timestamp_column] - cluster_start_time).total_seconds()
                if time_diff > time_threshold:
                    break
                current_activity = case_group.loc[current_idx, 'ProcessedActivity']
                if current_activity == base_activity:
                    cluster_indices.append(current_idx)
                else:
                    mismatch_count += 1
                    if mismatch_count <= max_mismatches:
                        cluster_indices.append(current_idx)
                    else:
                        break

            # Filter cluster to dominant base activity
            cluster_activities = case_group.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.mode().iloc[0]
            valid_cluster_indices = case_group.loc[cluster_indices][
                case_group.loc[cluster_indices, 'ProcessedActivity'] == dominant_activity].index.tolist()

            # Validate and mark cluster
            if len(valid_cluster_indices) >= min_matching_events:
                unsuffixed_indices = case_group.loc[valid_cluster_indices][
                    case_group.loc[valid_cluster_indices, 'BaseActivity'] == dominant_activity].index.tolist()
                if unsuffixed_indices:
                    keep_idx = unsuffixed_indices[0]
                else:
                    keep_idx = valid_cluster_indices[0]

                df.loc[valid_cluster_indices, 'is_collateral_event'] = 1
                df.loc[keep_idx, 'is_collateral_event'] = 0
                df.loc[valid_cluster_indices, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1

            i += len(cluster_indices)

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
        print("No labels available for metric calculation")

    # Step 7: Integrity Check
    total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
    total_marked = df['is_collateral_event'].sum()
    total_clean = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
    print(f"Total collateral clusters detected: {total_clusters}")
    print(f"Total events marked as collateral: {total_marked}")
    print(f"Events to be removed: {total_marked}")
    print(f"Clean events not modified: {total_clean}")

    # Step 8: Remove Collateral Events
    df_fixed = df[df['is_collateral_event'] == 0].copy()

    # Step 9: Save Output
    df_fixed['Activity_fixed'] = df_fixed[activity_column]
    df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
    if label_column in df.columns:
        output_columns.append(label_column)
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
    print(f"Error: {e}")