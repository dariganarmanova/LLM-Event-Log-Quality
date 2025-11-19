# Generated script for BPIC15-CollateralEvent - Run 2
# Generated on: 2025-11-18T21:21:26.929814
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column])

# Create iscollateral column
df['iscollateral'] = df[activity_column].apply(lambda x: 1 if isinstance(x, str) and x.endswith(collateral_suffix) else 0)

# Create BaseActivity column
def remove_collateral_suffix(act):
    if isinstance(act, str) and act.endswith(collateral_suffix):
        return act[:-len(collateral_suffix)]
    return act
df['BaseActivity'] = df[activity_column].apply(remove_collateral_suffix)

# Preprocess to create ProcessedActivity
def preprocess_activity(act):
    if not isinstance(act, str):
        return ''
    lower = act.lower()
    no_suffix = re.sub(activity_suffix_pattern, '', lower)
    normalized = no_suffix.replace('_', ' ').replace('-', ' ')
    normalized = re.sub(r'\s+', ' ', normalized.strip())
    return normalized
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_mask = df[case_column] == case
    case_df = df[case_mask].copy()
    n = len(case_df)
    i = 0
    while i < n:
        cluster_start_time = case_df.iloc[i][timestamp_column]
        base = case_df.iloc[i]['ProcessedActivity']
        cluster_local_indices = [i]
        mismatch_count = 0
        j = i + 1
        while j < n:
            time_diff = (case_df.iloc[j][timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            proc_j = case_df.iloc[j]['ProcessedActivity']
            if proc_j == base:
                cluster_local_indices.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_local_indices.append(j)
            j += 1
        # Process cluster
        cluster_procs = [case_df.iloc[k]['ProcessedActivity'] for k in cluster_local_indices]
        if cluster_procs:
            count = Counter(cluster_procs)
            dominant = count.most_common(1)[0][0]
            valid_local_indices = [k for k in cluster_local_indices if case_df.iloc[k]['ProcessedActivity'] == dominant]
            if len(valid_local_indices) >= min_matching_events:
                # Find keep_loc
                unsuffixed_locs = []
                for loc in valid_local_indices:
                    base_act = case_df.iloc[loc]['BaseActivity']
                    if isinstance(base_act, str) and not re.search(activity_suffix_pattern, base_act):
                        unsuffixed_locs.append(loc)
                if unsuffixed_locs:
                    keep_loc = min(unsuffixed_locs)
                else:
                    keep_loc = valid_local_indices[0]
                # Mark events
                for loc in valid_local_indices:
                    orig_idx = case_df.index[loc]
                    df.loc[orig_idx, 'CollateralGroup'] = cluster_counter
                    if loc == keep_loc:
                        df.loc[orig_idx, 'is_collateral_event'] = 0
                    else:
                        df.loc[orig_idx, 'is_collateral_event'] = 1
                cluster_counter += 1
                i = j
                continue
        i += 1

# Calculate Detection Metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(lab):
        if pd.isna(lab):
            return 0
        s = str(lab).lower()
        if 'collateral' in s:
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

# Integrity Check
num_clusters = cluster_counter
total_marked_collateral = (df['is_collateral_event'] == 1).sum()
events_to_remove = total_marked_collateral
clean_not_modified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {total_marked_collateral}")
print(f"Events to be removed: {events_to_remove}")
print(f"Clean events that were NOT modified: {clean_not_modified}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Prepare Output
columns_to_keep = [case_column, activity_column, timestamp_column]
optional_cols = ['Variant', 'Resource']
if label_column in df.columns:
    optional_cols.append(label_column)
for col in optional_cols:
    if col in df.columns:
        columns_to_keep.append(col)
df_output = df_fixed[columns_to_keep].copy()
df_output['Activity_fixed'] = df_output[activity_column]
df_output[timestamp_column] = df_output[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save Output
df_output.to_csv(output_file, index=False)

# Summary Statistics
original_events = len(df)
final_events = len(df_fixed)
removed = original_events - final_events
percentage = (removed / original_events * 100) if original_events > 0 else 0
unique_before = df[activity_column].nunique()
unique_after = df_fixed[activity_column].nunique()
print(f"Total events (original): {original_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_before}")
print(f"Unique activities after removal: {unique_after}")
print(f"Output file path: {output_file}")
if removed > 0:
    removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
    print("\nSample of removed events:")
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")

# Final Prints
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: collateral")