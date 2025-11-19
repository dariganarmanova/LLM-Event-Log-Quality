# Generated script for Credit-Collateral - Run 1
# Generated on: 2025-11-18T18:58:30.594650
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'

df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

has_label = label_column in df.columns

df[timestamp_column] = pd.to_datetime(df[timestamp_column])
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column]
mask = df['iscollateral'] == 1
df.loc[mask, 'BaseActivity'] = df.loc[mask, activity_column].str[:-len(collateral_suffix)]

def preprocess_activity(act):
    if pd.isna(act):
        return ''
    lower = str(act).lower()
    base = re.sub(activity_suffix_pattern, '', lower)
    base = re.sub(r'[_-]', ' ', base)
    base = ' '.join(base.split())
    return base

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

df['CollateralGroup'] = -1
df['is_collateral_event'] = 0

cluster_counter = 0

for case_name, group in df.groupby(case_column):
    case_df = group.copy().reset_index(drop=True)
    i = 0
    while i < len(case_df):
        start_time = case_df.at[i, timestamp_column]
        base_activity = case_df.at[i, 'ProcessedActivity']
        if pd.isna(base_activity) or base_activity == '':
            i += 1
            continue
        mismatch_count = 0
        cluster_indices = [i]
        j = i + 1
        while j < len(case_df):
            curr_time = case_df.at[j, timestamp_column]
            time_diff = (curr_time - start_time).total_seconds()
            if time_diff > time_threshold:
                break
            curr_proc = case_df.at[j, 'ProcessedActivity']
            if pd.isna(curr_proc) or curr_proc == '':
                j += 1
                continue
            if curr_proc == base_activity:
                cluster_indices.append(j)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_indices.append(j)
            j += 1
        cluster_procs = [case_df.at[k, 'ProcessedActivity'] for k in cluster_indices]
        if not cluster_procs:
            i += 1
            continue
        dominant = Counter(cluster_procs).most_common(1)[0][0]
        final_cluster = [k for k in cluster_indices if case_df.at[k, 'ProcessedActivity'] == dominant]
        if len(final_cluster) >= min_matching_events:
            unsuffixed = []
            for k in final_cluster:
                base_act_lower = str(case_df.at[k, 'BaseActivity']).lower()
                if not re.search(activity_suffix_pattern, base_act_lower):
                    unsuffixed.append(k)
            cluster_counter += 1
            group_id = cluster_counter
            if unsuffixed:
                for k in final_cluster:
                    case_df.at[k, 'is_collateral_event'] = 0 if k in unsuffixed else 1
                    case_df.at[k, 'CollateralGroup'] = group_id
            else:
                first_k = min(final_cluster)
                for k in final_cluster:
                    case_df.at[k, 'is_collateral_event'] = 0 if k == first_k else 1
                    case_df.at[k, 'CollateralGroup'] = group_id
        i = j
    df.loc[group.index, 'CollateralGroup'] = case_df['CollateralGroup'].values
    df.loc[group.index, 'is_collateral_event'] = case_df['is_collateral_event'].values

if has_label:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        str_label = str(label).lower()
        return 1 if 'collateral' in str_label else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_met = "✓" if precision >= 0.6 else "✗"
    status = "met" if prec_met == "✓" else "not met"
    print(f"{prec_met} Precision threshold (≥ 0.6) {status}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

num_clusters = df[df['CollateralGroup'] != -1]['CollateralGroup'].nunique()
num_marked_collateral = (df['is_collateral_event'] == 1).sum()
num_events_to_remove = num_marked_collateral
num_clean_preserved = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {num_clusters}")
print(f"Total events marked as collateral: {num_marked_collateral}")
print(f"Events to be removed: {num_events_to_remove}")

df_fixed = df[df['is_collateral_event'] == 0].copy()
df_fixed['Activity_fixed'] = df_fixed[activity_column]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

output_df = df_fixed[[case_column, timestamp_column, activity_column, 'Activity_fixed']].copy()
if 'Variant' in df_fixed.columns:
    output_df.insert(2, 'Variant', df_fixed['Variant'])
if has_label:
    output_df[label_column] = df_fixed[label_column]
if 'Resource' in df_fixed.columns:
    output_df['Resource'] = df_fixed['Resource']

output_df.to_csv(output_file, index=False)

original_events = len(df)
final_events = len(output_df)
removed = original_events - final_events
percentage = (removed / original_events * 100) if original_events > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = output_df[activity_column].nunique()
print(f"Total events (original): {original_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")

removed_df = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
print("Sample of removed events:")
print(removed_df.to_string(index=False))

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {output_df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: collateral")