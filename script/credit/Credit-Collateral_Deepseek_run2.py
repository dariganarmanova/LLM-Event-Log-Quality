# Generated script for Credit-Collateral - Run 2
# Generated on: 2025-11-13T16:05:43.862470
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
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

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names if needed
df.columns = df.columns.str.strip()
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Convert timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by case and timestamp
df.sort_values([case_column, timestamp_column], inplace=True)

# Create iscollateral column
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)

# Create BaseActivity column (remove collateral suffix)
df['BaseActivity'] = df[activity_column].replace(collateral_suffix, '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower() if not case_sensitive else str(activity)
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[-_]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case].copy()
    case_indices = case_df.index.tolist()
    
    i = 0
    while i < len(case_indices):
        start_idx = case_indices[i]
        start_time = df.at[start_idx, timestamp_column]
        base_activity = df.at[start_idx, 'ProcessedActivity']
        cluster = [start_idx]
        mismatch_count = 0
        
        j = i + 1
        while j < len(case_indices):
            current_idx = case_indices[j]
            current_time = df.at[current_idx, timestamp_column]
            time_diff = (current_time - start_time).total_seconds()
            
            if time_diff > time_threshold:
                break
                
            current_activity = df.at[current_idx, 'ProcessedActivity']
            if current_activity == base_activity:
                cluster.append(current_idx)
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster.append(current_idx)
                else:
                    break
            j += 1
        
        # Filter cluster to dominant base activity
        if len(cluster) > 0:
            cluster_activities = df.loc[cluster, 'ProcessedActivity']
            dominant_activity = cluster_activities.value_counts().idxmax()
            filtered_cluster = [idx for idx in cluster if df.at[idx, 'ProcessedActivity'] == dominant_activity]
            
            # Validate cluster size
            if len(filtered_cluster) >= min_matching_events:
                # Mark events for removal
                has_unsuffixed = False
                for idx in filtered_cluster:
                    original_activity = df.at[idx, activity_column]
                    processed = preprocess_activity(original_activity)
                    if original_activity.lower() == processed.lower():
                        has_unsuffixed = True
                        keep_idx = idx
                        break
                
                if has_unsuffixed:
                    # Keep unsuffixed version
                    for idx in filtered_cluster:
                        if idx == keep_idx:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                        df.at[idx, 'CollateralGroup'] = cluster_counter
                else:
                    # Keep first event chronologically
                    first_idx = filtered_cluster[0]
                    for idx in filtered_cluster:
                        if idx == first_idx:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                        df.at[idx, 'CollateralGroup'] = cluster_counter
                
                cluster_counter += 1
        
        # Move to next unclustered event
        if len(cluster) > 0:
            i = j
        else:
            i += 1

# Calculate detection metrics if label column exists
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        if isinstance(label, str) and 'collateral' in label.lower():
            return 1
        return 0
    
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Integrity check
collateral_clusters = df[df['CollateralGroup'] != -1]['CollateralGroup'].nunique()
events_marked = df['is_collateral_event'].sum()
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {collateral_clusters}")
print(f"Total events marked as collateral: {events_marked}")
print(f"Clean events not modified: {clean_events}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
output_columns = [case_column, timestamp_column, activity_column]
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
if label_column in df.columns:
    output_columns.append(label_column)

df_fixed['Activity_fixed'] = df_fixed[activity_column]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, columns=output_columns, index=False)

# Summary statistics
original_count = len(df)
processed_count = len(df_fixed)
removed_count = original_count - processed_count
removed_pct = (removed_count / original_count) * 100
unique_before = df[activity_column].nunique()
unique_after = df_fixed[activity_column].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {processed_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {unique_before}")
print(f"Unique activities after removal: {unique_after}")
print(f"Output file path: {output_file}")

# Print sample of removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)
if not removed_events.empty:
    print("\nSample of removed events:")
    print(removed_events[[case_column, activity_column, timestamp_column]].to_string(index=False))

# Print required run information
print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: collateral")