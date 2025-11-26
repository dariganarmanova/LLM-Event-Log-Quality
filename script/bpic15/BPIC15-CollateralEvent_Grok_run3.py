# Generated script for BPIC15-CollateralEvent - Run 3
# Generated on: 2025-11-18T21:22:38.835792
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Handle column naming variations for Case
if 'Case ID' in df.columns:
    df[case_column] = df['Case ID']
    df = df.drop('Case ID', axis=1)

# Ensure required columns exist (assume they do, but check)
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# #2. Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.rstrip(collateral_suffix)

# #3. Preprocess Activity Names
def normalize_activity(act):
    if pd.isna(act):
        return ''
    act_lower = str(act).lower()
    act_norm = re.sub(r'[_-]', ' ', act_lower)
    act_norm = re.sub(r'\s+', ' ', act_norm).strip()
    return act_norm

def remove_suffixes(norm_act):
    if pd.isna(norm_act):
        return ''
    return re.sub(activity_suffix_pattern, '', norm_act)

df['Activity_for_proc'] = df['BaseActivity']
df['NormalizedActivity'] = df['Activity_for_proc'].apply(normalize_activity)
df['ProcessedActivity'] = df['NormalizedActivity'].apply(remove_suffixes)

# #4. Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# #5. Sliding Window Clustering
i = 0
while i < len(df):
    current_case = df.loc[i, case_column]
    case_start = i
    case_end = i
    while case_end < len(df) and df.loc[case_end, case_column] == current_case:
        case_end += 1
    k = case_start
    while k < case_end:
        if df.loc[k, 'CollateralGroup'] != -1:
            k += 1
            continue
        # Start new cluster
        cluster_local_indices = [k]
        start_time = df.loc[k, timestamp_column]
        base_activity = df.loc[k, 'ProcessedActivity']
        mismatch_count = 0
        m = k + 1
        while m < case_end:
            if df.loc[m, 'CollateralGroup'] != -1:
                m += 1
                continue
            time_diff = (df.loc[m, timestamp_column] - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            proc_act = df.loc[m, 'ProcessedActivity']
            if proc_act == base_activity:
                cluster_local_indices.append(m)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_local_indices.append(m)
            m += 1
        # Filter to dominant base
        if cluster_local_indices:
            cluster_procs = [df.loc[idx, 'ProcessedActivity'] for idx in cluster_local_indices]
            dominant = Counter(cluster_procs).most_common(1)[0][0]
            final_cluster_indices = [idx for idx in cluster_local_indices if df.loc[idx, 'ProcessedActivity'] == dominant]
            if len(final_cluster_indices) >= min_matching_events:
                # Valid cluster, mark events
                unsuffixed_in_cluster = [idx for idx in final_cluster_indices if df.loc[idx, 'NormalizedActivity'] == df.loc[idx, 'ProcessedActivity']]
                if unsuffixed_in_cluster:
                    keep_idx = min(unsuffixed_in_cluster)
                else:
                    keep_idx = min(final_cluster_indices)
                for idx in final_cluster_indices:
                    df.loc[idx, 'CollateralGroup'] = cluster_counter
                    if idx == keep_idx:
                        df.loc[idx, 'is_collateral_event'] = 0
                    else:
                        df.loc[idx, 'is_collateral_event'] = 1
                cluster_counter += 1
        k += 1
    i = case_end

# #7. Calculate Detection Metrics (BEFORE REMOVAL)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower()
        if 'collateral' in label_str:
            return 1
        return 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8. Integrity Check
num_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
num_marked_collateral = (df['is_collateral_event'] == 1).sum()
num_removed = num_marked_collateral
num_clean_unmodified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {num_marked_collateral}")
print(f"Events to be removed: {num_removed}")
print(f"Clean events that were NOT modified: {num_clean_unmodified}")

# #9. Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# #10. Save Output
# Drop tracking columns
tracking_cols = ['iscollateral', 'BaseActivity', 'NormalizedActivity', 'ProcessedActivity', 'CollateralGroup', 'is_collateral_event', 'Activity_for_proc']
df_fixed = df_fixed.drop(columns=[col for col in tracking_cols if col in df_fixed.columns], errors='ignore')
# Add Activity_fixed
df_fixed['Activity_fixed'] = df_fixed[activity_column]
# Format timestamp
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
# Save
df_fixed.to_csv(output_file, index=False)

# #11. Summary Statistics
original_count = len(df)
after_count = len(df_fixed)
removed_count = original_count - after_count
percentage = (removed_count / original_count * 100) if original_count > 0 else 0
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {after_count}")
print(f"Events removed: {removed_count} ({percentage:.2f}%)")
print(f"Unique activities before removal: {df[activity_column].nunique()}")
print(f"Unique activities after removal: {df_fixed[activity_column].nunique()}")
print(f"Output file path: {output_file}")
if removed_count > 0:
    print("\nSample of up to 10 removed events:")
    removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
    removed_sample[timestamp_column] = removed_sample[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")

print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: collateral")