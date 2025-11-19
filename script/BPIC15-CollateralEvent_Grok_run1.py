# Generated script for BPIC15-CollateralEvent - Run 1
# Generated on: 2025-11-18T21:20:33.008325
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'
time_threshold = 2.0
max_mismatches = 1
min_matching_events = 2
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
case_sensitive = False

print(f"Run 1: Original dataset shape: {pd.read_csv(input_file).shape}")

# Step 1: Load CSV
df = pd.read_csv(input_file)

# Normalize column names
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': 'Case'})
if 'Complete Timestamp' in df.columns:
    df = df.rename(columns={'Complete Timestamp': 'Timestamp'})
if 'Activity' not in df.columns:
    raise ValueError("Activity column not found")
if 'Case' not in df.columns:
    raise ValueError("Case column not found")
if 'Timestamp' not in df.columns:
    raise ValueError("Timestamp column not found")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by Case and Timestamp
df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

# Step 2: Identify Collateral Activities
df['is_collateral'] = df['Activity'].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(r':collateral$', '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    act = str(activity).lower() if not case_sensitive else str(activity)
    act = re.sub(activity_suffix_pattern, '', act)
    act = re.sub(r'[_-]', ' ', act)
    act = re.sub(r'\s+', ' ', act).strip()
    return act

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Helper function for unsuffixed
def is_unsuffixed(base_act):
    if pd.isna(base_act):
        return True
    cleaned = re.sub(activity_suffix_pattern, '', str(base_act))
    return cleaned == str(base_act)

# Step 5: Sliding Window Clustering
for case_name, case_group in df.groupby('Case'):
    case_indices = case_group.index.tolist()
    i = 0
    while i < len(case_indices):
        start_idx = case_indices[i]
        cluster_start_time = df.at[start_idx, 'Timestamp']
        base_activity = df.at[start_idx, 'ProcessedActivity']
        cluster_events = [start_idx]
        mismatch_count = 0
        last_j = i
        for j in range(i + 1, len(case_indices)):
            curr_idx = case_indices[j]
            time_diff = (df.at[curr_idx, 'Timestamp'] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            curr_proc = df.at[curr_idx, 'ProcessedActivity']
            if curr_proc == base_activity:
                cluster_events.append(curr_idx)
                last_j = j
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster_events.append(curr_idx)
                    last_j = j
                else:
                    break
        # Filter to dominant base
        cluster_procs = [df.at[idx, 'ProcessedActivity'] for idx in cluster_events]
        proc_counter = Counter(cluster_procs)
        if proc_counter:
            dominant_base = proc_counter.most_common(1)[0][0]
            filtered_cluster = [idx for idx in cluster_events if df.at[idx, 'ProcessedActivity'] == dominant_base]
            if len(filtered_cluster) >= min_matching_events:
                # Assign group to filtered cluster
                for idx in filtered_cluster:
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                # Mark events for removal
                unsuffixed_indices = [idx for idx in filtered_cluster if is_unsuffixed(df.at[idx, 'BaseActivity'])]
                if unsuffixed_indices:
                    for idx in filtered_cluster:
                        df.at[idx, 'is_collateral_event'] = 0 if is_unsuffixed(df.at[idx, 'BaseActivity']) else 1
                else:
                    keep_idx = filtered_cluster[0]  # First chronologically
                    for idx in filtered_cluster:
                        df.at[idx, 'is_collateral_event'] = 0 if idx == keep_idx else 1
                cluster_counter += 1
                # Advance i past the window
                i = last_j + 1
                continue
        # If not valid, advance to next
        i += 1

# Step 7: Calculate Detection Metrics (BEFORE REMOVAL)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower()
        return 1 if 'collateral' in label_str else 0
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

# Step 8: Integrity Check
total_clusters = len(df[df['CollateralGroup'] >= 0]['CollateralGroup'].unique()) if (df['CollateralGroup'] >= 0).any() else 0
events_marked_collateral = (df['is_collateral_event'] == 1).sum()
events_to_remove = events_marked_collateral
clean_events_unchanged = ((df['is_collateral'] == 0) & (df['is_collateral_event'] == 0)).sum()
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {events_marked_collateral}")
print(f"Events to be removed: {events_to_remove}")
print(f"Clean events not modified: {clean_events_unchanged}")

# Step 9: Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Step 10: Save Output
# Drop processing columns
columns_to_drop = ['is_collateral', 'BaseActivity', 'ProcessedActivity', 'CollateralGroup', 'is_collateral_event']
if 'y_true' in df_fixed.columns:
    columns_to_drop.append('y_true')
for col in columns_to_drop:
    if col in df_fixed.columns:
        df_fixed.drop(col, axis=1, inplace=True)
df_fixed['Activity_fixed'] = df_fixed['Activity']
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_original = len(df)
total_after = len(df_fixed)
removed_count = total_original - total_after
percentage = (removed_count / total_original * 100) if total_original > 0 else 0
unique_acts_before = df['Activity'].nunique()
unique_acts_after = df_fixed['Activity'].nunique()
print(f"Total events (original): {total_original}")
print(f"Total events (after removal): {total_after}")
print(f"Events removed: {removed_count} ({percentage:.2f}%)")
print(f"Unique activities before removal: {unique_acts_before}")
print(f"Unique activities after removal: {unique_acts_after}")
print(f"Output file path: {output_file}")
print("Sample removed events:")
removed_sample = df[df['is_collateral_event'] == 1][['Case', 'Activity', 'Timestamp']].head(10)
if not removed_sample.empty:
    print(removed_sample.to_string(index=False))
else:
    print("No removed events")

print(f"Run 1: Processed dataset saved to: data/bpic15/bpic15_collateral_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: collateral")