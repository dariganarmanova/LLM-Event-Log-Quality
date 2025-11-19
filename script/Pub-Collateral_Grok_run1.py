# Generated script for Pub-Collateral - Run 1
# Generated on: 2025-11-18T18:35:12.171516
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
max_mismatches = 1
min_matching_events = 2
activity_suffix_pattern = r"(_signed\d*|_\d+)$"

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize Case column if needed
if 'CaseID' in df.columns and case_column not in df.columns:
    df[case_column] = df.pop('CaseID')

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{collateral_suffix}$', '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(act):
    if pd.isna(act):
        return act
    act_lower = act.lower()
    act_no_suffix = re.sub(activity_suffix_pattern, '', act_lower)
    act_norm = re.sub(r'[_-]', ' ', act_no_suffix)
    act_norm = re.sub(r'\s+', ' ', act_norm).strip()
    return act_norm

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Step 5: Sliding Window Clustering
for case_name, group in df.groupby(case_column, sort=False):
    group_indices = group.index.tolist()
    i = 0
    while i < len(group_indices):
        cluster = [group_indices[i]]
        start_time = df.loc[group_indices[i], timestamp_column]
        base_activity = df.loc[group_indices[i], 'ProcessedActivity']
        mismatch_count = 0
        j = i + 1
        while j < len(group_indices):
            curr_idx = group_indices[j]
            time_diff = (df.loc[curr_idx, timestamp_column] - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            proc_act = df.loc[curr_idx, 'ProcessedActivity']
            if pd.isna(proc_act) or pd.isna(base_activity):
                j += 1
                continue
            if proc_act == base_activity:
                cluster.append(curr_idx)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster.append(curr_idx)
            j += 1
        # Filter to dominant base activity
        cluster_procs = [df.loc[idx, 'ProcessedActivity'] for idx in cluster if pd.notna(df.loc[idx, 'ProcessedActivity'])]
        if not cluster_procs:
            i = j
            continue
        dominant = Counter(cluster_procs).most_common(1)[0][0]
        filtered_cluster = [idx for idx in cluster if df.loc[idx, 'ProcessedActivity'] == dominant]
        if len(filtered_cluster) >= min_matching_events:
            # Valid cluster
            # Find unsuffixed events
            unsuffixed_indices = []
            for idx in filtered_cluster:
                base_act = df.loc[idx, 'BaseActivity']
                if pd.notna(base_act) and not re.search(activity_suffix_pattern, base_act):
                    unsuffixed_indices.append(idx)
            if unsuffixed_indices:
                # Keep the earliest unsuffixed
                keep_idx = min(unsuffixed_indices, key=lambda x: df.loc[x, timestamp_column])
                for f_idx in filtered_cluster:
                    if f_idx != keep_idx:
                        df.loc[f_idx, 'is_collateral_event'] = 1
            else:
                # Keep the first chronologically
                keep_idx = filtered_cluster[0]
                for f_idx in filtered_cluster:
                    if f_idx != keep_idx:
                        df.loc[f_idx, 'is_collateral_event'] = 1
            # Assign cluster ID
            for f_idx in filtered_cluster:
                df.loc[f_idx, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1
        i = j

# Step 7: Calculate Detection Metrics (BEFORE REMOVAL)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'collateral' in str(label).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_met = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_met} Precision threshold (≥ 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
num_clusters = cluster_counter
events_marked = (df['is_collateral_event'] == 1).sum()
events_removed = events_marked
clean_not_modified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {events_marked}")
print(f"Events to be removed: {events_removed}")
print(f"Clean events not modified: {clean_not_modified}")

# Step 9: Remove Collateral Events
added_cols = ['BaseActivity', 'ProcessedActivity', 'iscollateral', 'CollateralGroup', 'is_collateral_event']
original_columns = [col for col in df.columns if col not in added_cols]
df_fixed = df[df['is_collateral_event'] == 0][original_columns].copy()

# Step 10: Save Output
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_original = len(df)
total_after = len(df_fixed)
removed_count = total_original - total_after
percentage = (removed_count / total_original * 100) if total_original > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = df_fixed[activity_column].nunique()
print(f"Total events (original): {total_original}")
print(f"Total events (after removal): {total_after}")
print(f"Events removed: {removed_count} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
print("Sample removed events:")
removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
if not removed_sample.empty:
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")
print(f"Run 1: Processed dataset saved to: data/pub/pub_collateral_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: collateral")