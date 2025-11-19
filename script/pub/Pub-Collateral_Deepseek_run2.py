# Generated script for Pub-Collateral - Run 2
# Generated on: 2025-11-13T17:41:52.839212
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
import re
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
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

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names if needed
df.columns = df.columns.str.strip()
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Convert timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by Case and Timestamp
df.sort_values([case_column, timestamp_column], inplace=True)

# Step 2: Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(collateral_suffix, '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[-_]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Step 5: Sliding Window Clustering
for case in df[case_column].unique():
    case_events = df[df[case_column] == case].copy()
    indices = case_events.index.tolist()
    n_events = len(indices)
    i = 0
    
    while i < n_events:
        current_idx = indices[i]
        if df.at[current_idx, 'CollateralGroup'] != -1:
            i += 1
            continue
            
        cluster_start_time = df.at[current_idx, timestamp_column]
        base_activity = df.at[current_idx, 'ProcessedActivity']
        cluster_indices = [current_idx]
        mismatch_count = 0
        j = i + 1
        
        while j < n_events:
            next_idx = indices[j]
            time_diff = (df.at[next_idx, timestamp_column] - cluster_start_time).total_seconds()
            
            if time_diff > time_threshold:
                break
                
            next_activity = df.at[next_idx, 'ProcessedActivity']
            if next_activity == base_activity:
                cluster_indices.append(next_idx)
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster_indices.append(next_idx)
                else:
                    break
            j += 1
        
        # Filter cluster to dominant base activity
        if len(cluster_indices) > 0:
            activities_in_cluster = df.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = activities_in_cluster.value_counts().idxmax()
            filtered_cluster = [idx for idx in cluster_indices if df.at[idx, 'ProcessedActivity'] == dominant_activity]
            
            if len(filtered_cluster) >= min_matching_events:
                # Mark events for removal
                unsuffixed_exists = False
                for idx in filtered_cluster:
                    original_activity = df.at[idx, 'BaseActivity']
                    processed = preprocess_activity(original_activity)
                    if original_activity.lower() == processed.lower():
                        unsuffixed_exists = True
                        break
                
                if unsuffixed_exists:
                    for idx in filtered_cluster:
                        original_activity = df.at[idx, 'BaseActivity']
                        processed = preprocess_activity(original_activity)
                        if original_activity.lower() == processed.lower():
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                else:
                    df.at[filtered_cluster[0], 'is_collateral_event'] = 0
                    for idx in filtered_cluster[1:]:
                        df.at[idx, 'is_collateral_event'] = 1
                
                # Assign cluster ID
                for idx in filtered_cluster:
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1
        
        i = j

# Step 7: Calculate Detection Metrics (if labels exist)
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

# Step 8: Integrity Check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_marked = df['is_collateral_event'].sum()
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_marked}")
print(f"Clean events not modified: {clean_events}")

# Step 9: Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Step 10: Save Output
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

# Step 11: Summary Statistics
original_count = len(df)
final_count = len(df_fixed)
removed_count = original_count - final_count
removed_pct = (removed_count / original_count) * 100

unique_activities_before = df[activity_column].nunique()
unique_activities_after = df_fixed[activity_column].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {final_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {unique_activities_before}")
print(f"Unique activities after removal: {unique_activities_after}")
print(f"Output file path: {output_file}")

# Print sample of removed events
removed_events = df[df['is_collateral_event'] == 1]
if not removed_events.empty:
    sample = removed_events.head(min(10, len(removed_events)))
    print("\nSample of removed events:")
    for _, row in sample.iterrows():
        print(f"Case: {row[case_column]}, Activity: {row[activity_column]}, Timestamp: {row[timestamp_column]}")

# REQUIRED: Print summary
print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: