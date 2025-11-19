# Generated script for Pub-Collateral - Run 2
# Generated on: 2025-11-14T13:23:58.934644
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
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run2.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File {input_file} not found.")
    exit(1)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
optional_columns = ['Variant', 'Resource', 'label']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Missing required column '{col}' in the input file.")
        exit(1)

# Normalize timestamp column
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)
except Exception as e:
    print(f"Error processing timestamps: {e}")
    exit(1)

# Create helper columns
df['iscollateral'] = df['Activity'].str.endswith(':collateral').astype(int)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['BaseActivity'].str.lower() if not case_sensitive else df['BaseActivity']
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_\-\s]+', ' ', regex=True).str.strip()
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0

# Sliding window clustering
cluster_counter = 0
for case_id, group in df.groupby('Case'):
    group = group.reset_index()
    n = len(group)
    i = 0

    while i < n:
        cluster_start_time = group.loc[i, 'Timestamp']
        base_activity = group.loc[i, 'ProcessedActivity']
        cluster_indices = [i]
        mismatch_count = 0

        for j in range(i + 1, n):
            time_diff = (group.loc[j, 'Timestamp'] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break

            if group.loc[j, 'ProcessedActivity'] == base_activity:
                cluster_indices.append(j)
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster_indices.append(j)
                else:
                    break

        # Validate cluster
        if len(cluster_indices) >= min_matching_events:
            cluster_activities = group.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.mode().iloc[0]
            valid_cluster = group.loc[cluster_indices]
            valid_indices = valid_cluster[valid_cluster['ProcessedActivity'] == dominant_activity].index

            # Mark for removal
            unsuffixed_event = valid_cluster[valid_cluster['Activity'] == valid_cluster['BaseActivity']]
            if not unsuffixed_event.empty:
                keep_index = unsuffixed_event.index[0]
            else:
                keep_index = valid_cluster.iloc[0].name

            group.loc[valid_indices, 'is_collateral_event'] = 1
            group.loc[keep_index, 'is_collateral_event'] = 0
            group.loc[cluster_indices, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

        i = cluster_indices[-1] + 1

    df.loc[group['index'], ['is_collateral_event', 'CollateralGroup']] = group[['is_collateral_event', 'CollateralGroup']].values

# Calculate detection metrics
if 'label' in df.columns:
    df['y_true'] = df['label'].fillna('').str.contains('collateral', case=False).astype(int)
    y_true = df['y_true']
    y_pred = df['is_collateral_event']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("\n=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("\nNo labels available for metric calculation.")
    precision = recall = f1 = 0.0

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
events_marked = df['is_collateral_event'].sum()
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {events_marked}")
print(f"Events to be removed: {events_marked}")
print(f"Clean events (not modified): {clean_events}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
df_fixed['Activity_fixed'] = df_fixed['Activity']
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
output_columns = ['Case', 'Timestamp', 'Variant', 'Activity', 'Activity_fixed', 'label']
output_columns = [col for col in output_columns if col in df_fixed.columns]
df_fixed.to_csv(output_file, index=False)

# Summary statistics
total_events = len(df)
final_events = len(df_fixed)
removed_events = total_events - final_events
removed_percentage = (removed_events / total_events) * 100
unique_activities_before = df['Activity'].nunique()
unique_activities_after = df_fixed['Activity'].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {total_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed_events} ({removed_percentage:.2f}%)")
print(f"Unique activities before removal: {unique_activities_before}")
print(f"Unique activities after removal: {unique_activities_after}")
print(f"Run 2: Processed dataset saved to: {output_file}")