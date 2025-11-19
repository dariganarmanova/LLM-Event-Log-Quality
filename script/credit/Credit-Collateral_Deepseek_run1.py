# Generated script for Credit-Collateral - Run 1
# Generated on: 2025-11-13T16:04:46.890552
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run1.csv'
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
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found in input file")

# Convert timestamp to datetime
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(collateral_suffix, '', regex=True)

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[-_]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
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
                if mismatch_count <= max_mismatches:
                    cluster_events.append(next_event)
                    cluster_indices.append(case_indices[j])
                else:
                    break
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
            
            # Validate cluster size
            if len(filtered_cluster) >= min_matching_events:
                # Mark events for removal
                has_unsuffixed = False
                unsuffixed_index = -1
                for idx, event in zip(filtered_indices, filtered_cluster):
                    original_activity = event['BaseActivity']
                    processed_activity = event['ProcessedActivity']
                    if re.sub(activity_suffix_pattern, '', original_activity.lower()).strip() == processed_activity:
                        has_unsuffixed = True
                        unsuffixed_index = idx
                        break
                
                if has_unsuffixed:
                    # Keep unsuffixed event, remove others
                    for idx in filtered_indices:
                        if idx == unsuffixed_index:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                else:
                    # Keep first event, remove others
                    first_index = filtered_indices[0]
                    for idx in filtered_indices:
                        if idx == first_index:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                
                # Assign cluster ID
                for idx in filtered_indices:
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                cluster_counter += 1
        
        i = j

# Calculate Detection Metrics
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

# Integrity Check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_marked = df['is_collateral_event'].sum()
clean_events = ((df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)).sum()

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_marked}")
print(f"Clean events not modified: {clean_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save Output
output_columns = [case_column, timestamp_column, activity_column]
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
if label_column in df.columns:
    output_columns.append(label_column)

df_fixed['Activity_fixed'] = df_fixed[activity_column]
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed[output_columns].to_csv(output_file, index=False)

# Summary Statistics
original_count = df.shape[0]
processed_count = df_fixed.shape[0]
removed_count = original_count - processed_count
removed_pct = (removed_count / original_count) * 100

original_activities = df[activity_column].nunique()
processed_activities = df_fixed[activity_column].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {processed_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {original_activities}")
print(f"Unique activities after removal: {processed_activities}")
print(f"Output file path: {output_file}")

removed_events = df[df['is_collateral_event'] == 1]
print("\nSample of removed events (up to 10):")
print(removed_events.head(10)[[case_column, activity_column, timestamp_column]])

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed