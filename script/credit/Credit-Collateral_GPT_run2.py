# Generated script for Credit-Collateral - Run 2
# Generated on: 2025-11-13T15:09:19.066725
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from datetime import timedelta
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
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run2.csv'

# Column names
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = [case_column, activity_column, timestamp_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert Timestamp to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    if df[timestamp_column].isnull().any():
        raise ValueError("Invalid timestamps detected in the dataset.")

    # Sort by Case and Timestamp
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

    df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

    # Step 4: Initialize Tracking Columns
    df['CollateralGroup'] = -1
    df['is_collateral_event'] = 0
    cluster_counter = 0

    # Step 5: Sliding Window Clustering
    for case_id, case_data in df.groupby(case_column):
        case_data = case_data.reset_index()
        n = len(case_data)
        i = 0

        while i < n:
            cluster_start_time = case_data.loc[i, timestamp_column]
            base_activity = case_data.loc[i, 'ProcessedActivity']
            cluster_indices = [i]
            mismatch_count = 0

            for j in range(i + 1, n):
                time_diff = (case_data.loc[j, timestamp_column] - cluster_start_time).total_seconds()
                if time_diff > time_threshold:
                    break

                if case_data.loc[j, 'ProcessedActivity'] == base_activity:
                    cluster_indices.append(j)
                else:
                    mismatch_count += 1
                    if mismatch_count <= max_mismatches:
                        cluster_indices.append(j)
                    else:
                        break

            # Filter cluster to dominant base activity
            cluster_activities = case_data.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.mode()[0]
            valid_cluster_indices = case_data.index[cluster_indices][
                cluster_activities == dominant_activity
            ]

            # Validate cluster
            if len(valid_cluster_indices) >= min_matching_events:
                df.loc[valid_cluster_indices, 'CollateralGroup'] = cluster_counter
                unsuffixed_events = case_data.loc[valid_cluster_indices, activity_column].apply(
                    lambda x: preprocess_activity(x) == x
                )

                if unsuffixed_events.any():
                    keep_index = valid_cluster_indices[unsuffixed_events.idxmax()]
                else:
                    keep_index = valid_cluster_indices[0]

                df.loc[valid_cluster_indices, 'is_collateral_event'] = 1
                df.loc[keep_index, 'is_collateral_event'] = 0
                cluster_counter += 1

            i = cluster_indices[-1] + 1

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
        print("No labels available for metric calculation.")

    # Step 7: Integrity Check
    total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
    total_removed = df['is_collateral_event'].sum()
    total_clean = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])

    print(f"Total collateral clusters detected: {total_clusters}")
    print(f"Total events marked as collateral: {total_removed}")
    print(f"Clean events not modified: {total_clean}")

    # Step 8: Remove Collateral Events
    df_fixed = df[df['is_collateral_event'] == 0].copy()

    # Step 9: Save Output
    df_fixed['Activity_fixed'] = df_fixed[activity_column]
    df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_fixed.to_csv(output_file, index=False)

    # Step 10: Summary Statistics
    print(f"Run 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {df_fixed.shape}")
    print(f"Run 2: Total events removed: {total_removed} ({(total_removed / len(df)) * 100:.2f}%)")
    print(f"Run 2: Unique activities before removal: {df[activity_column].nunique()}")
    print(f"Run 2: Unique activities after removal: {df_fixed[activity_column].nunique()}")
    print("Sample of removed events:")
    print(df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10))

except Exception as e:
    print(f"An error occurred: {e}")