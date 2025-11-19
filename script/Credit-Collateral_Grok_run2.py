# Generated script for Credit-Collateral - Run 2
# Generated on: 2025-11-18T18:59:32.444429
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run2.csv'
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

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load CSV
df = pd.read_csv(input_file)

# Normalize CaseID to Case if needed
if 'CaseID' in df.columns and case_column not in df.columns:
    df = df.rename(columns={'CaseID': case_column})

# Ensure required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Optional columns
has_label = label_column in df.columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column])

# Step 2: Identify Collateral Activities
df['BaseActivity'] = df[activity_column].str.rstrip(collateral_suffix)
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)

# Define preprocess function
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    act = str(activity).rstrip(':collateral')
    match = re.search(activity_suffix_pattern + '$', act)
    if match:
        act = act[:match.start()]
    act = act.lower()
    act = re.sub(r'[_-]', ' ', act)
    act = re.sub(r'\s+', ' ', act.strip())
    return act

# Step 3: Preprocess Activity Names
df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Step 4: Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0

# Define is_unsuffixed function
def is_unsuffixed(activity):
    if pd.isna(activity):
        return False
    act = str(activity).rstrip(':collateral')
    return not bool(re.search(activity_suffix_pattern + '$', act))

# Step 5: Sliding Window Clustering
cluster_counter = 0
for case_id in df[case_column].unique():
    mask = df[case_column] == case_id
    df_case = df.loc[mask].copy()
    df_case = df_case.sort_values(timestamp_column)
    orig_indices = df_case.index.values
    df_case = df_case.reset_index(drop=True)
    n = len(df_case)
    clustered = set()
    i = 0
    while i < n:
        if i in clustered:
            i += 1
            continue
        start_time = df_case.iloc[i][timestamp_column]
        base_activity = df_case.iloc[i]['ProcessedActivity']
        cluster = [i]
        mismatch_count = 0
        j = i + 1
        while j < n:
            if j in clustered:
                j += 1
                continue
            time_diff = (df_case.iloc[j][timestamp_column] - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            proc_act = df_case.iloc[j]['ProcessedActivity']
            if proc_act == base_activity:
                cluster.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster.append(j)
            j += 1
        # Filter to dominant
        cluster_acts = [df_case.iloc[k]['ProcessedActivity'] for k in cluster]
        act_counts = Counter(cluster_acts)
        if act_counts:
            dominant = act_counts.most_common(1)[0][0]
            filtered = [k for k in cluster if df_case.iloc[k]['ProcessedActivity'] == dominant]
            if len(filtered) >= min_matching_events:
                # Valid cluster
                unsuffixed = [k for k in filtered if is_unsuffixed(df_case.iloc[k][activity_column])]
                if unsuffixed:
                    keep_local = min(unsuffixed)
                else:
                    keep_local = min(filtered)
                for k in filtered:
                    global_idx = orig_indices[k]
                    if k == keep_local:
                        df.loc[global_idx, 'is_collateral_event'] = 0
                    else:
                        df.loc[global_idx, 'is_collateral_event'] = 1
                    df.loc[global_idx, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1
                clustered.update(filtered)
        i += 1

# Step 7: Calculate Detection Metrics (BEFORE REMOVAL)
print("=== Detection Performance Metrics ===")
if has_label:
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

# Step 8: Integrity Check
total_clusters = cluster_counter
events_marked = (df['is_collateral_event'] == 1).sum()
clean_unmodified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {events_marked}")
print(f"Events to be removed: {events_marked}")
print(f"Unmodified clean events: {clean_unmodified}")

# Step 9: Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Step 10: Save Output
output_columns = [case_column, timestamp_column, activity_column]
if has_variant:
    output_columns.append('Variant')
if has_resource:
    output_columns.append('Resource')
if has_label:
    output_columns.append(label_column)
df_fixed['Activity_fixed'] = df_fixed[activity_column]
output_columns.append('Activity_fixed')
df_out = df_fixed[output_columns].copy()
df_out[timestamp_column] = df_out[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_out.to_csv(output_file, index=False)

# Step 11: Summary Statistics
original_events = len(df)
final_events = len(df_fixed)
removed = original_events - final_events
percentage = (removed / original_events * 100) if original_events > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = df_fixed[activity_column].nunique()
print(f"Total events (original): {original_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
print("Sample of removed events:")
removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
if len(removed_sample) > 0:
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_out.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: collateral")