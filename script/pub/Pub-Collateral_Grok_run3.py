# Generated script for Pub-Collateral - Run 3
# Generated on: 2025-11-18T18:37:41.148969
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load CSV
df = pd.read_csv(input_file)

# Normalize column names if needed
column_map = {'CaseID': 'Case', 'Case ID': 'Case', 'case': 'Case'}
for old, new in column_map.items():
    if old in df.columns:
        df = df.rename(columns={old: new})

# Ensure required columns exist (assume they do after normalization)
df[timestamp_column] = pd.to_datetime(df[timestamp_column])
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{collateral_suffix}$', '', regex=True)

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower() if not case_sensitive else df['BaseActivity']
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_-]', ' ', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case_val, case_mask in df.groupby(case_column):
    orig_indices = df[case_mask].index.tolist()
    sorted_orig = sorted(orig_indices, key=lambda x: df.at[x, timestamp_column])
    i = 0
    n = len(sorted_orig)
    while i < n:
        curr_idx = sorted_orig[i]
        if df.at[curr_idx, 'CollateralGroup'] != -1:
            i += 1
            continue
        cluster_start_time = df.at[curr_idx, timestamp_column]
        base_activity = df.at[curr_idx, 'ProcessedActivity']
        cluster_local_pos = [i]
        mismatch_count = 0
        j = i + 1
        expanded_max_local = i
        while j < n:
            next_idx = sorted_orig[j]
            time_diff = (df.at[next_idx, timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            next_proc = df.at[next_idx, 'ProcessedActivity']
            if next_proc == base_activity:
                cluster_local_pos.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                else:
                    cluster_local_pos.append(j)
            expanded_max_local = j
            j += 1
        expanded_max_local = j - 1 if j > i + 1 else i
        expanded_indices = [sorted_orig[p] for p in cluster_local_pos]
        proc_acts = [df.at[idx, 'ProcessedActivity'] for idx in expanded_indices]
        count = Counter(proc_acts)
        if not count:
            i = expanded_max_local + 1
            continue
        dominant = count.most_common(1)[0][0]
        filtered_local = [p for p in cluster_local_pos if df.at[sorted_orig[p], 'ProcessedActivity'] == dominant]
        filtered_indices = [sorted_orig[p] for p in filtered_local]
        if len(filtered_indices) >= min_matching_events:
            unsuffixed_exists = False
            for f_idx in filtered_indices:
                base_act = df.at[f_idx, 'BaseActivity']
                if re.search(activity_suffix_pattern, base_act):
                    df.at[f_idx, 'is_collateral_event'] = 1
                else:
                    unsuffixed_exists = True
                    df.at[f_idx, 'is_collateral_event'] = 0
            if not unsuffixed_exists:
                df.at[filtered_indices[0], 'is_collateral_event'] = 0
                for k in range(1, len(filtered_indices)):
                    df.at[filtered_indices[k], 'is_collateral_event'] = 1
            global cluster_counter
            cluster_counter += 1
            for f_idx in filtered_indices:
                df.at[f_idx, 'CollateralGroup'] = cluster_counter
        i = expanded_max_local + 1

# Calculate Detection Metrics (BEFORE REMOVAL)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'collateral' in str(label).lower() else 0
    df['y_true'] = df[label_column].apply(normalize_label)
    y_true = df['y_true']
    y_pred = df['is_collateral_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_met = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_met} Precision threshold (≥ 0.6) met/not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Integrity Check
total_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
events_marked_collateral = (df['is_collateral_event'] == 1).sum()
events_removed = events_marked_collateral
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {events_marked_collateral}")
print(f"Events to be removed: {events_removed}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()
df_fixed['Activity_fixed'] = df_fixed[activity_column]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Drop temporary columns
drops = ['BaseActivity', 'ProcessedActivity', 'iscollateral', 'CollateralGroup', 'is_collateral_event']
if 'y_true' in df_fixed.columns:
    drops.append('y_true')
df_fixed = df_fixed.drop(columns=[col for col in drops if col in df_fixed.columns], errors='ignore')

# Save Output
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
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

# Sample of up to 10 removed events
if events_removed > 0:
    removed_df = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
    removed_df[timestamp_column] = removed_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    print("Sample removed events:")
    print(removed_df.to_string(index=False))
else:
    print("Sample removed events: None")

print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: collateral")