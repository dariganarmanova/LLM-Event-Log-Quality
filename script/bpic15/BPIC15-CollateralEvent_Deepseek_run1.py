# Generated script for BPIC15-CollateralEvent - Run 1
# Generated on: 2025-11-13T14:45:28.965687
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run1.csv'
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

# Normalize column names if needed
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Convert timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by case and timestamp
df.sort_values([case_column, timestamp_column], inplace=True)

# Identify collateral activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '', regex=False)

# Preprocess activity names
df['ProcessedActivity'] = df[activity_column].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_-]', ' ', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()

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
        current_idx = case_indices[i]
        if df.at[current_idx, 'CollateralGroup'] != -1:
            i += 1
            continue
            
        cluster_start_time = df.at[current_idx, timestamp_column]
        base_activity = df.at[current_idx, 'ProcessedActivity']
        cluster_indices = [current_idx]
        mismatch_count = 0
        
        j = i + 1
        while j < len(case_indices):
            next_idx = case_indices[j]
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
        
        # Filter to dominant base activity
        if len(cluster_indices) > 0:
            activities = df.loc[cluster_indices, 'ProcessedActivity'].value_counts()
            dominant_activity = activities.idxmax()
            filtered_indices = [idx for idx in cluster_indices if df.at[idx, 'ProcessedActivity'] == dominant_activity]
            
            if len(filtered_indices) >= min_matching_events:
                # Mark events for removal
                has_unsuffixed = False
                for idx in filtered_indices:
                    original_activity = df.at[idx, activity_column].replace(collateral_suffix, '')
                    processed = re.sub(activity_suffix_pattern, '', original_activity.lower())
                    processed = re.sub(r'[_-]', ' ', processed)
                    processed = re.sub(r'\s+', ' ', processed).strip()
                    if processed == df.at[idx, 'ProcessedActivity']:
                        has_unsuffixed = True
                        keep_idx = idx
                        break
                
                if has_unsuffixed:
                    for idx in filtered_indices:
                        if idx == keep_idx:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                        df.at[idx, 'CollateralGroup'] = cluster_counter
                else:
                    first_idx = filtered_indices[0]
                    for idx in filtered_indices:
                        if idx == first_idx:
                            df.at[idx, 'is_collateral_event'] = 0
                        else:
                            df.at[idx, 'is_collateral_event'] = 1
                        df.at[idx, 'CollateralGroup'] = cluster_counter
                
                cluster_counter += 1
                i = j
            else:
                i += 1
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
marked_for_removal = df['is_collateral_event'].sum()
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])

print("\n=== Integrity Check ===")
print(f"Total collateral clusters detected: {collateral_clusters}")
print(f"Total events marked as collateral: {marked_for_removal}")
print(f"Clean events not modified: {clean_events}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
output_columns = [case_column, timestamp_column, activity_column]
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

removed_samples = df[df['is_collateral_event'] == 1].head(10)
if not removed_samples.empty:
    print("\nSample of removed events:")
    print(removed_samples[[case_column, activity_column, timestamp_column]].to_string(index=False))

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: collateral")