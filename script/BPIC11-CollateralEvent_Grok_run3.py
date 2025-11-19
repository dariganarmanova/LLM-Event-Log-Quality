# Generated script for BPIC11-CollateralEvent - Run 3
# Generated on: 2025-11-18T22:09:08.795897
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter

# Configuration
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
max_mismatches = 1
min_matching_events = 2
activity_suffix_pattern = r"(_signed\d*|_\d+)$"

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load CSV
df = pd.read_csv(input_file)
df[timestamp_column] = pd.to_datetime(df[timestamp_column])
df = df.sort_values([case_column, timestamp_column])

# Handle optional columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns
has_label = label_column in df.columns

# #2. Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)

def get_base_activity(act):
    if pd.isna(act):
        return act
    act_str = str(act)
    if act_str.endswith(collateral_suffix):
        return act_str[:-len(collateral_suffix)]
    return act_str

df['BaseActivity'] = df[activity_column].apply(get_base_activity)

# #3. Preprocess Activity Names
def preprocess_activity(act):
    if pd.isna(act):
        return act
    act_lower = str(act).lower()
    act_no_suffix = re.sub(activity_suffix_pattern, '', act_lower)
    act_normalized = re.sub(r'[_-]', ' ', act_no_suffix)
    act_normalized = re.sub(r'\s+', ' ', act_normalized).strip()
    return act_normalized

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# #4. Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# #5. Sliding Window Clustering
def is_unsuffixed(act):
    if pd.isna(act):
        return False
    act_str = str(act)
    base = act_str.rstrip(collateral_suffix) if act_str.endswith(collateral_suffix) else act_str
    cleaned = re.sub(activity_suffix_pattern, '', base)
    return cleaned == base

for case_name, group in df.groupby(case_column):
    if len(group) < min_matching_events:
        continue
    group = group.sort_values(timestamp_column)
    i = 0
    while i < len(group):
        row_i = group.iloc[i]
        cluster_start_time = row_i[timestamp_column]
        base_activity = row_i['ProcessedActivity']
        if pd.isna(base_activity):
            i += 1
            continue
        mismatch_count = 0
        window_end = i
        for j in range(i + 1, len(group)):
            row_j = group.iloc[j]
            event_time = row_j[timestamp_column]
            time_diff = (event_time - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            processed_j = row_j['ProcessedActivity']
            if pd.isna(processed_j):
                window_end = j
                continue
            if processed_j == base_activity:
                window_end = j
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                window_end = j
        if window_end == i:
            i += 1
            continue
        window_df = group.iloc[i:window_end + 1]
        act_counts = Counter(window_df['ProcessedActivity'].dropna())
        if not act_counts:
            i += 1
            continue
        dominant = max(act_counts, key=act_counts.get)
        cluster_df = window_df[window_df['ProcessedActivity'] == dominant]
        cluster_size = len(cluster_df)
        if cluster_size >= min_matching_events:
            unsuffixed_mask = cluster_df[activity_column].apply(is_unsuffixed)
            unsuffixed_df = cluster_df[unsuffixed_mask]
            if len(unsuffixed_df) > 0:
                keep_orig_idx = unsuffixed_df.index[0]
            else:
                keep_orig_idx = cluster_df.index[0]
            current_group = cluster_counter
            for orig_idx in cluster_df.index:
                df.loc[orig_idx, 'CollateralGroup'] = current_group
                if orig_idx != keep_orig_idx:
                    df.loc[orig_idx, 'is_collateral_event'] = 1
            cluster_counter += 1
        i = window_end + 1

# #7. Calculate Detection Metrics (BEFORE REMOVAL)
if has_label:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower()
        return 1 if 'collateral' in label_str else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        prec_threshold_met = "✓" if prec >= 0.6 else "✗"
        print(f"{prec_threshold_met} Precision threshold (≥ 0.6) met/not met")
    except:
        print("=== Detection Performance Metrics ===")
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")
        print("Error in metric calculation")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8. Integrity Check
num_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
num_marked_removal = (df['is_collateral_event'] == 1).sum()
clean_not_modified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {num_marked_removal}")
print(f"Events to be removed: {num_marked_removal}")
print(f"Clean events that were NOT modified: {clean_not_modified}")

# #9. Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()
original_total = len(df)

# #10. Save Output
df_fixed['Activity_fixed'] = df_fixed[activity_column]
columns_to_keep = [case_column, 'Activity_fixed', timestamp_column]
if has_variant:
    columns_to_keep.append('Variant')
if has_resource:
    columns_to_keep.append('Resource')
if has_label:
    columns_to_keep.append(label_column)
out_df = df_fixed[columns_to_keep].copy()
out_df[timestamp_column] = out_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
out_df.to_csv(output_file, index=False)

# #11. Summary Statistics
print(f"Total events (original): {original_total}")
print(f"Total events (after removal): {len(out_df)}")
removed_count = original_total - len(out_df)
removed_pct = (removed_count / original_total * 100) if original_total > 0 else 0
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
unique_acts_before = df[activity_column].nunique()
unique_acts_after = out_df['Activity_fixed'].nunique()
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
print("Sample of up to 10 removed events:")
print(removed_sample.to_string(index=False))

print(f"Run 3: Processed dataset saved to: data/bpic11/bpic11_collateral_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {out_df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: collateral")