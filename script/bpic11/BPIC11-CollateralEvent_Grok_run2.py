# Generated script for BPIC11-CollateralEvent - Run 2
# Generated on: 2025-11-18T22:08:06.991203
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter

# Configuration
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run2.csv'
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

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load CSV
df = pd.read_csv(input_file)

# Normalize CaseID to Case if needed
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Convert Timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Create iscollateral column
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix, na=False).astype(int)

# Create BaseActivity column: remove collateral suffix
df['BaseActivity'] = df[activity_column].str.rsplit(collateral_suffix, n=1).str[0]

# Preprocess function
def preprocess_activity(activity):
    if pd.isna(activity):
        return activity
    activity = str(activity).lower() if not case_sensitive else str(activity)
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[_-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

# Create ProcessedActivity from BaseActivity
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Function to check if unsuffixed
def is_unsuffixed(base_act, processed):
    if pd.isna(base_act) or pd.isna(processed):
        return False
    norm_base = re.sub(r'[_-]', ' ', str(base_act).lower())
    norm_base = re.sub(r'\s+', ' ', norm_base).strip()
    return norm_base == processed

# Sliding Window Clustering
for case_name, case_df in df.groupby(case_column):
    i = 0
    while i < len(case_df):
        cluster_events = [i]
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
                cluster_events.append(j)
                j += 1
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_events.append(j)
                j += 1
        # Process cluster
        if len(cluster_events) >= min_matching_events:
            cluster_proc_acts = [case_df.iloc[k]['ProcessedActivity'] for k in cluster_events]
            dominant_count = Counter(cluster_proc_acts)
            if dominant_count:
                dominant = dominant_count.most_common(1)[0][0]
                filtered_cluster = [k for k in cluster_events if case_df.iloc[k]['ProcessedActivity'] == dominant]
                if len(filtered_cluster) >= min_matching_events:
                    # Determine keep
                    unsuffixed_events = [k for k in filtered_cluster if is_unsuffixed(case_df.iloc[k]['BaseActivity'], case_df.iloc[k]['ProcessedActivity'])]
                    if unsuffixed_events:
                        keep_idx = min(unsuffixed_events)
                    else:
                        keep_idx = min(filtered_cluster)
                    # Mark events
                    for k in filtered_cluster:
                        orig_idx = case_df.index[k]
                        if k == keep_idx:
                            df.loc[orig_idx, 'is_collateral_event'] = 0
                        else:
                            df.loc[orig_idx, 'is_collateral_event'] = 1
                    # Assign group
                    for k in filtered_cluster:
                        orig_idx = case_df.index[k]
                        df.loc[orig_idx, 'CollateralGroup'] = cluster_counter
                    cluster_counter += 1
        # Advance i
        i = j

# Calculate Detection Metrics (BEFORE REMOVAL)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower()
        return 1 if 'collateral' in label_str else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    from sklearn.metrics import precision_score, recall_score, f1_score
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
events_marked = (df['is_collateral_event'] == 1).sum()
events_removed = events_marked
clean_not_modified = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {events_marked}")
print(f"Events to be removed: {events_removed}")
print(f"Clean events not modified: {clean_not_modified}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Prepare output
df_fixed['Activity_fixed'] = df_fixed[activity_column]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns
output_cols = [case_column, activity_column, 'Activity_fixed', timestamp_column]
for col in ['Variant', 'Resource', label_column]:
    if col in df_fixed.columns:
        output_cols.append(col)
output_df = df_fixed[output_cols]

# Save Output
output_df.to_csv(output_file, index=False)

# Summary Statistics
original_events = len(df)
final_events = len(output_df)
removed_count = original_events - final_events
percentage = (removed_count / original_events * 100) if original_events > 0 else 0
unique_acts_before = df[activity_column].nunique()
unique_acts_after = output_df[activity_column].nunique()
print(f"Total events (original): {original_events}")
print(f"Total events (after removal): {final_events}")
print(f"Events removed: {removed_count} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")

# Sample removed events
print("\nSample of up to 10 removed events:")
removed_sample = df[df['is_collateral_event'] == 1][[case_column, activity_column, timestamp_column]].head(10)
if len(removed_sample) > 0:
    print(removed_sample.to_string(index=False))
else:
    print("No events removed.")

print(f"Run 2: Processed dataset saved to: data/bpic11/bpic11_collateral_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {output_df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: collateral")