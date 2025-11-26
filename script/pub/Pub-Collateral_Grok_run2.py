# Generated script for Pub-Collateral - Run 2
# Generated on: 2025-11-18T18:36:17.999623
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run2.csv'
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
if 'CaseID' in df.columns:
    df[case_column] = df['CaseID']
    df = df.drop('CaseID', axis=1)

# Ensure required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column])

# #2. Identify Collateral Activities
df['iscollateral'] = 0
mask = df[activity_column].str.endswith(collateral_suffix, na=False)
df.loc[mask, 'iscollateral'] = 1
df['BaseActivity'] = df[activity_column]
df.loc[mask, 'BaseActivity'] = df.loc[mask, activity_column].str[:-len(collateral_suffix)]

# #3. Preprocess Activity Names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    act_lower = str(activity).lower()
    act_no_suffix = re.sub(activity_suffix_pattern, '', act_lower)
    act_norm = re.sub(r'[_-]', ' ', act_no_suffix)
    act_norm = ' '.join(act_norm.split())
    return act_norm

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# #4. Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# #5. Sliding Window Clustering
for case_id, case_df in df.groupby(case_column):
    n = len(case_df)
    if n < min_matching_events:
        continue
    i = 0
    case_indices = case_df.index.tolist()
    while i < n:
        cluster_rel = [i]
        start_time = case_df.iloc[i][timestamp_column]
        base_act = case_df.iloc[i]['ProcessedActivity']
        mismatch_cnt = 0
        j = i + 1
        while j < n:
            time_diff = (case_df.iloc[j][timestamp_column] - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            proc_j = case_df.iloc[j]['ProcessedActivity']
            if proc_j == base_act:
                cluster_rel.append(j)
            else:
                mismatch_cnt += 1
                if mismatch_cnt > max_mismatches:
                    break
                cluster_rel.append(j)
            j += 1
        # Filter to dominant
        cluster_acts = [case_df.iloc[k]['ProcessedActivity'] for k in cluster_rel]
        dominant = Counter(cluster_acts).most_common(1)[0][0]
        final_rel = [k for k in cluster_rel if case_df.iloc[k]['ProcessedActivity'] == dominant]
        if len(final_rel) >= min_matching_events:
            # Valid cluster, mark
            unsuffixed_rel = []
            for k in final_rel:
                orig_base = case_df.iloc[k]['BaseActivity']
                if not re.search(activity_suffix_pattern, str(orig_base), re.IGNORECASE):
                    unsuffixed_rel.append(k)
            if unsuffixed_rel:
                keep_rel = min(unsuffixed_rel)
            else:
                keep_rel = final_rel[0]
            for k in final_rel:
                df_idx = case_indices[k]
                df.loc[df_idx, 'CollateralGroup'] = cluster_counter
                if k == keep_rel:
                    df.loc[df_idx, 'is_collateral_event'] = 0
                else:
                    df.loc[df_idx, 'is_collateral_event'] = 1
            cluster_counter += 1
            i = j
        else:
            i += 1

# #7. Calculate Detection Metrics (BEFORE REMOVAL)
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

# #8. Integrity Check
num_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
events_marked_collateral = (df['is_collateral_event'] == 1).sum()
events_to_remove = events_marked_collateral
clean_not_modified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {events_marked_collateral}")
print(f"Events to be removed: {events_to_remove}")
print(f"Clean events that were NOT modified: {clean_not_modified}")

# #9. Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# #10. Save Output
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed['Activity_fixed'] = df_fixed[activity_column]
cols_to_drop = ['iscollateral', 'BaseActivity', 'ProcessedActivity', 'CollateralGroup', 'is_collateral_event']
for col in cols_to_drop:
    if col in df_fixed.columns:
        df_fixed = df_fixed.drop(col, axis=1)
df_fixed.to_csv(output_file, index=False)

# #11. Summary Statistics
total_original = len(df)
total_after = len(df_fixed)
removed = total_original - total_after
percentage = (removed / total_original * 100) if total_original > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = df_fixed[activity_column].nunique()
print(f"Total events (original): {total_original}")
print(f"Total events (after removal): {total_after}")
print(f"Events removed: {removed} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
print("Sample removed events:")
if removed > 0:
    removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")

print(f"Run 2: Processed dataset saved to: data/pub/pub_collateral_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: collateral")