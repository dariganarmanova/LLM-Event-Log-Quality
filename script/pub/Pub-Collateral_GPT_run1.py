# Generated script for Pub-Collateral - Run 1
# Generated on: 2025-11-14T13:23:27.267856
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

# File paths
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run1.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
optional_columns = ['Variant', 'Resource', 'label']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Missing required column '{col}' in input file.")
        exit()

# Normalize column names
df.rename(columns=lambda x: x.strip(), inplace=True)

# Convert Timestamp to datetime
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
except Exception as e:
    print(f"Error parsing timestamps: {e}")
    exit()

# Drop rows with invalid timestamps
df.dropna(subset=['Timestamp'], inplace=True)

# Sort data by Case and Timestamp
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify collateral activities
df['iscollateral'] = df['Activity'].str.endswith(':collateral').astype(int)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '', regex=True)

# Preprocess activity names
df['ProcessedActivity'] = df['BaseActivity'].str.lower() if not case_sensitive else df['BaseActivity']
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_\-\s]+', ' ', regex=True).str.strip()

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case_id, case_group in df.groupby('Case'):
    case_group = case_group.reset_index()
    n = len(case_group)
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue

        cluster = [i]
        cluster_start_time = case_group.loc[i, 'Timestamp']
        base_activity = case_group.loc[i, 'ProcessedActivity']
        mismatch_count = 0

        for j in range(i + 1, n):
            time_diff = (case_group.loc[j, 'Timestamp'] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break

            if case_group.loc[j, 'ProcessedActivity'] == base_activity:
                cluster.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster.append(j)

        # Filter cluster to dominant base activity
        cluster_activities = case_group.loc[cluster, 'ProcessedActivity']
        dominant_activity = cluster_activities.mode().iloc[0]
        cluster = [idx for idx in cluster if case_group.loc[idx, 'ProcessedActivity'] == dominant_activity]

        if len(cluster) >= min_matching_events:
            # Mark events in the cluster
            unsuffixed_events = [idx for idx in cluster if case_group.loc[idx, 'Activity'] == case_group.loc[idx, 'BaseActivity']]
            if unsuffixed_events:
                keep_idx = unsuffixed_events[0]
            else:
                keep_idx = cluster[0]

            for idx in cluster:
                if idx != keep_idx:
                    case_group.loc[idx, 'is_collateral_event'] = 1

            case_group.loc[cluster, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

        visited = [visited[k] or (k in cluster) for k in range(n)]

    # Update the main DataFrame
    df.loc[case_group['index'], 'is_collateral_event'] = case_group['is_collateral_event']
    df.loc[case_group['index'], 'CollateralGroup'] = case_group['CollateralGroup']

# Detection metrics
if 'label' in df.columns:
    df['y_true'] = df['label'].str.contains('collateral', case=False, na=False).astype(int)
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

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_removed = df['is_collateral_event'].sum()
total_clean = len(df) - total_removed
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_removed}")
print(f"Events to be removed: {total_removed}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save the cleaned dataset
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Total events (original): {len(df)}")
print(f"Run 1: Total events (after removal): {len(df_fixed)}")
print(f"Run 1: Events removed: {total_removed} ({(total_removed / len(df)) * 100:.2f}%)")
print(f"Run 1: Unique activities before removal: {df['Activity'].nunique()}")
print(f"Run 1: Unique activities after removal: {df_fixed['Activity'].nunique()}")