# Generated script for BPIC11-CollateralEvent - Run 1
# Generated on: 2025-11-18T22:07:15.042787
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
max_mismatches = 1
min_matching_events = 2
activity_suffix_pattern = r'(_signed\d*|_\d+)$'

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Check required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns
has_label = label_column in df.columns

# Convert Timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# #2 Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '', regex=False)

# #3 Preprocess Activity Names
def preprocess_activity(activity, pattern, collateral_suffix, case_sensitive=False):
    if activity.endswith(collateral_suffix):
        base = activity[:-len(collateral_suffix)]
    else:
        base = activity
    low = base.lower() if not case_sensitive else base
    proc = re.sub(pattern, '', low)
    proc = re.sub(r'[_-]', ' ', proc)
    proc = re.sub(r'\s+', ' ', proc).strip()
    return proc

df['ProcessedActivity'] = df[activity_column].apply(lambda x: preprocess_activity(x, activity_suffix_pattern, collateral_suffix))

# #4 Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# #5 Sliding Window Clustering
def is_unsuffixed(activity, pattern, collateral_suffix):
    if activity.endswith(collateral_suffix):
        base = activity[:-len(collateral_suffix)]
    else:
        base = activity
    low = base.lower()
    cleaned = re.sub(pattern, '', low)
    return cleaned == low

for case_id, case_df in df.groupby(case_column):
    clustered = set()
    for i in range(len(case_df)):
        if i in clustered:
            continue
        cluster_pos = [i]
        start_time = case_df.iloc[i][timestamp_column]
        base_activity = case_df.iloc[i]['ProcessedActivity']
        mismatch_count = 0
        j = i + 1
        while j < len(case_df):
            event_time = case_df.iloc[j][timestamp_column]
            time_diff = (event_time - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            event_proc = case_df.iloc[j]['ProcessedActivity']
            if event_proc == base_activity:
                cluster_pos.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_pos.append(j)
            j += 1
        # Filter to dominant base
        cluster_procs = [case_df.iloc[p]['ProcessedActivity'] for p in cluster_pos]
        count = Counter(cluster_procs)
        if count:
            dominant = count.most_common(1)[0][0]
            valid_pos = [p for p in cluster_pos if case_df.iloc[p]['ProcessedActivity'] == dominant]
            if len(valid_pos) >= min_matching_events:
                # Determine keep
                candidates = []
                for p in valid_pos:
                    ts = case_df.iloc[p][timestamp_column]
                    act = case_df.iloc[p][activity_column]
                    uns = is_unsuffixed(act, activity_suffix_pattern, collateral_suffix)
                    candidates.append((ts, p, uns))
                candidates.sort(key=lambda x: x[0])
                has_uns = any(c[2] for c in candidates)
                if has_uns:
                    keep_p = next(p for _, p, uns in candidates if uns)
                else:
                    keep_p = candidates[0][1]
                # Mark
                for p in valid_pos:
                    orig_idx = case_df.iloc[p].name
                    if p != keep_p:
                        df.loc[orig_idx, 'is_collateral_event'] = 1
                    df.loc[orig_idx, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1
                clustered.update(valid_pos)

# #7 Calculate Detection Metrics (BEFORE REMOVAL)
if has_label:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        str_label = str(label).lower()
        return 1 if 'collateral' in str_label else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) {'met' if prec >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8 Integrity Check
num_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
num_marked_collateral = (df['is_collateral_event'] == 1).sum()
num_events_to_remove = num_marked_collateral
num_clean_untouched = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {num_marked_collateral}")
print(f"Events to be removed: {num_events_to_remove}")
print(f"Clean events not modified: {num_clean_untouched}")

# #9 Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# #10 Save Output
df_fixed['Activity_fixed'] = df_fixed[activity_column]
out_cols = [case_column, timestamp_column]
if has_variant:
    out_cols.append('Variant')
if has_resource:
    out_cols.append('Resource')
out_cols += [activity_column, 'Activity_fixed']
if has_label:
    out_cols.append(label_column)
df_fixed = df_fixed[out_cols]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# #11 Summary Statistics
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
removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
print("Sample of removed events:")
print(removed_sample.to_string(index=False))
print(f"Run 1: Processed dataset saved to: data/bpic11/bpic11_collateral_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: collateral")