# Generated script for BPIC11-CollateralEvent - Run 2
# Generated on: 2025-11-13T11:35:11.302904
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
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
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the input file")

# Convert timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Step 2: Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(collateral_suffix, '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[-_]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Step 5: Sliding Window Clustering
for case in df[case_column].unique():
    case_events = df[df[case_column] == case].copy()
    case_indices = case_events.index.tolist()
    
    i = 0
    while i < len(case_events):
        current_event = case_events.iloc[i]
        cluster_start_time = current_event[timestamp_column]
        base_activity = current_event['ProcessedActivity']
        cluster_events = [current_event]
        cluster_indices = [case_indices[i]]
        mismatch_count = 0
        
        j = i + 1
        while j < len(case_events):
            next_event = case_events.iloc[j]
            time_diff = (next_event[timestamp_column] - cluster_start_time).total_seconds()
            
            if time_diff > time_threshold:
                break
                
            if next_event['ProcessedActivity'] == base_activity:
                cluster_events.append(next_event)
                cluster_indices.append(case_indices[j])
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_events.append(next_event)
                cluster_indices.append(case_indices[j])
            
            j += 1
        
        # Filter cluster to dominant base activity
        if len(cluster_events) > 0:
            activity_counts = pd.Series([e['ProcessedActivity'] for e in cluster_events]).value_counts()
            dominant_activity = activity_counts.idxmax()
            filtered_cluster = []
            filtered_indices = []
            for idx, event in zip(cluster_indices, cluster_events):
                if event['ProcessedActivity'] == dominant_activity:
                    filtered_cluster.append(event)
                    filtered_indices.append(idx)
            
            # Validate cluster
            if len(filtered_cluster) >= min_matching_events:
                # Step 6: Mark Events for Removal
                has_unsuffixed = False
                for event in filtered_cluster:
                    original_activity = event['BaseActivity']
                    processed = preprocess_activity(original_activity)
                    if original_activity.lower() == processed.lower():
                        has_unsuffixed = True
                        break
                
                for idx, event in zip(filtered_indices, filtered_cluster):
                    original_activity = event['BaseActivity']
                    processed = preprocess_activity(original_activity)
                    if has_unsuffixed:
                        if original_activity.lower() == processed.lower():
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                    else:
                        if idx == filtered_indices[0]:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                    
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                
                cluster_counter += 1
        
        # Move to next unclustered event
        i = j

# Step 7: Calculate Detection Metrics
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        if 'collateral' in str(label).lower():
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

# Step 8: Integrity Check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].values else 0)
total_marked = df['is_collateral_event'].sum()
clean_events = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_marked}")
print(f"Clean events not modified: {clean_events}")

# Step 9: Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Step 10: Save Output
output_columns = [case_column, timestamp_column, activity_column]
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

original_activities = df[activity_column].nunique()
final_activities = df_fixed[activity_column].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {final_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {original_activities}")
print(f"Unique activities after removal: {final_activities}")
print(f"Output file path: {output_file}")

removed_events = df[df['is_collateral_event'] == 1].head(10)
if not removed_events.empty:
    print("\nSample of removed events:")
    print(removed_events[[case_column, activity_column, timestamp_column]].to_string(index=False))

print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task