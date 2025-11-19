# Generated script for Pub-Collateral - Run 1
# Generated on: 2025-11-13T17:40:35.021369
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/pub/Pub-Collateral.csv'
output_file = 'data/pub/pub_collateral_cleaned_run1.csv'
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

# Load data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names if needed
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Convert timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by case and timestamp
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Identify collateral activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(f'{collateral_suffix}$', '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower() if not case_sensitive else str(activity)
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[-_]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case in df[case_column].unique():
    case_events = df[df[case_column] == case].copy()
    case_indices = case_events.index.tolist()
    
    i = 0
    while i < len(case_events):
        current_idx = case_indices[i]
        current_event = case_events.loc[current_idx]
        cluster_start_time = current_event[timestamp_column]
        base_activity = current_event['ProcessedActivity']
        
        cluster_indices = [current_idx]
        mismatch_count = 0
        
        j = i + 1
        while j < len(case_events):
            next_idx = case_indices[j]
            next_event = case_events.loc[next_idx]
            time_diff = (next_event[timestamp_column] - cluster_start_time).total_seconds()
            
            if time_diff > time_threshold:
                break
                
            if next_event['ProcessedActivity'] == base_activity:
                cluster_indices.append(next_idx)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_indices.append(next_idx)
            
            j += 1
        
        # Filter to dominant base activity
        if len(cluster_indices) > 1:
            cluster_activities = df.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.value_counts().idxmax()
            filtered_indices = [idx for idx in cluster_indices if df.loc[idx, 'ProcessedActivity'] == dominant_activity]
            
            if len(filtered_indices) >= min_matching_events:
                # Mark events for removal
                cluster_activities = df.loc[filtered_indices, 'BaseActivity']
                unsuffixed_activities = [act for act in cluster_activities if not re.search(activity_suffix_pattern, act)]
                
                if unsuffixed_activities:
                    keep_idx = next(idx for idx in filtered_indices if df.loc[idx, 'BaseActivity'] == unsuffixed_activities[0])
                else:
                    keep_idx = filtered_indices[0]
                
                for idx in filtered_indices:
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                    df.at[idx, 'is_collateral_event'] = 0 if idx == keep_idx else 1
                
                cluster_counter += 1
                i = j
                continue
        
        i += 1

# Calculate metrics if label column exists
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'collateral' in str(label).lower() else 0
    
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
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].values else 0)
total_marked = df['is_collateral_event'].sum()
clean_events = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_marked}")
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
final_count = len(df_fixed)
removed_count = original_count - final_count
removed_pct = (removed_count / original_count) * 100
original_activities = df[activity_column].nunique()
final_activities = df_fixed[activity_column].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {final_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {original_activities}")
print(f"Unique activities after removal: {final_activities}")
print(f"Output file path: {output_file}")

# Sample of removed events
removed_events = df[df['is_collateral_event'] == 1]
if not removed_events.empty:
    sample_size = min(10, len(removed_events))
    print("\nSample of removed events:")
    print(removed_events.head(sample_size)[[case_column, activity_column, timestamp_column]].to_string(index=False))

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: collateral")