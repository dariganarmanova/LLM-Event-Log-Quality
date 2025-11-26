# Generated script for Credit-Collateral - Run 3
# Generated on: 2025-11-18T19:00:25.638651
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run3.csv'
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

# Handle column naming variations
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Check optional columns
has_label = label_column in df.columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns

# Store original columns
original_cols = list(df.columns)

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Create iscollateral column
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)

# Create BaseActivity column
df['BaseActivity'] = df[activity_column].str.rsplit(collateral_suffix, n=1).str[0]

# Preprocess Activity Names to create ProcessedActivity
def preprocess_activity(act):
    if pd.isna(act):
        return ''
    act_lower = act.lower()
    act_no_suffix = re.sub(activity_suffix_pattern, '', act_lower)
    act_normalized = re.sub(r'[_-]', ' ', act_no_suffix)
    act_normalized = re.sub(r'\s+', ' ', act_normalized).strip()
    return act_normalized

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case_val, group in df.groupby(case_column, sort=True):
    if len(group) < min_matching_events:
        continue
    group_sorted = group.sort_values(timestamp_column)
    orig_indices = group_sorted.index.tolist()
    i = 0
    while i < len(group_sorted):
        cluster = [i]
        start_time = group_sorted.iloc[i][timestamp_column]
        base_activity = group_sorted.iloc[i]['ProcessedActivity']
        mismatch_count = 0
        window_end = i + 1
        broke = False
        for j in range(i + 1, len(group_sorted)):
            time_diff = (group_sorted.iloc[j][timestamp_column] - start_time).total_seconds()
            if time_diff > time_threshold:
                window_end = j
                broke = True
                break
            proc_j = group_sorted.iloc[j]['ProcessedActivity']
            if proc_j == base_activity:
                cluster.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    window_end = j
                    broke = True
                    break
                cluster.append(j)
            window_end = j + 1
        if not broke:
            window_end = len(group_sorted)
        # Filter to dominant base
        cluster_procs = [group_sorted.iloc[k]['ProcessedActivity'] for k in cluster]
        if cluster_procs:
            dominant = Counter(cluster_procs).most_common(1)[0][0]
            valid_cluster_local = [k for k in cluster if group_sorted.iloc[k]['ProcessedActivity'] == dominant]
            if len(valid_cluster_local) >= min_matching_events:
                # Determine keep event
                candidates_unsuffixed = []
                for loc_i in valid_cluster_local:
                    act = group_sorted.iloc[loc_i][activity_column]
                    act_clean = act.rsplit(':', 1)[0] if act.endswith(':collateral') else act
                    flags = re.IGNORECASE if not case_sensitive else 0
                    if not re.search(activity_suffix_pattern + '$', act_clean, flags):
                        candidates_unsuffixed.append(loc_i)
                if candidates_unsuffixed:
                    keep_local = min(candidates_unsuffixed)
                else:
                    keep_local = min(valid_cluster_local)
                # Mark events
                for loc_i in valid_cluster_local:
                    orig_idx = orig_indices[loc_i]
                    df.loc[orig_idx, 'CollateralGroup'] = cluster_counter
                    if loc_i == keep_local:
                        df.loc[orig_idx, 'is_collateral_event'] = 0
                    else:
                        df.loc[orig_idx, 'is_collateral_event'] = 1
                cluster_counter += 1
        i = window_end

# 7. Calculate Detection Metrics (BEFORE REMOVAL)
print("=== Detection Performance Metrics ===")
if has_label:
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

# 8. Integrity Check
num_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
num_marked_collateral = (df['is_collateral_event'] == 1).sum()
num_events_to_remove = num_marked_collateral
num_clean_preserved = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {num_marked_collateral}")
print(f"Events to be removed: {num_events_to_remove}")
print(f"Clean events preserved: {num_clean_preserved}")

# 9. Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# 10. Save Output
# Format Timestamp
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
# Select original columns
df_output = df_fixed[original_cols].copy()
# Add Activity_fixed
activity_idx = df_output.columns.get_loc(activity_column)
df_output.insert(activity_idx + 1, 'Activity_fixed', df_output[activity_column])
df_output.to_csv(output_file, index=False)

# 11. Summary Statistics
original_events = len(df)
final_events = len(df_fixed)
removed_count = original_events - final_events
percentage_removed = (removed_count / original_events * 100) if original_events > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = df_fixed[activity_column].nunique()
print(f"Total events (original): {original_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed_count} ({percentage_removed:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
# Sample removed events
removed_df = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
print("\nSample removed events (up to 10):")
if not removed_df.empty:
    print(removed_df.to_string(index=False))
else:
    print("No events removed.")

print(f"Run 3: Processed dataset saved to: data/credit/credit_collateral_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: collateral")