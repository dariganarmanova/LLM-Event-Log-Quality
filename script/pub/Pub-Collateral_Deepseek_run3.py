# Generated script for Pub-Collateral - Run 3
# Generated on: 2025-11-13T17:42:50.296809
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv('data/pub/Pub-Collateral.csv')
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in input file")

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# Sort by Case and Timestamp
df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

# Identify collateral activities
collateral_suffix = ':collateral'
df['iscollateral'] = df['Activity'].str.endswith(collateral_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(collateral_suffix, '', regex=False)

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
for case in df['Case'].unique():
    case_df = df[df['Case'] == case].copy()
    case_indices = case_df.index.tolist()
    
    i = 0
    while i < len(case_indices):
        current_idx = case_indices[i]
        if df.at[current_idx, 'CollateralGroup'] != -1:
            i += 1
            continue
            
        cluster_start_time = df.at[current_idx, 'Timestamp']
        base_activity = df.at[current_idx, 'ProcessedActivity']
        cluster_indices = [current_idx]
        mismatch_count = 0
        
        j = i + 1
        while j < len(case_indices):
            next_idx = case_indices[j]
            time_diff = (df.at[next_idx, 'Timestamp'] - cluster_start_time).total_seconds()
            
            if time_diff > time_threshold:
                break
                
            next_activity = df.at[next_idx, 'ProcessedActivity']
            if next_activity == base_activity:
                cluster_indices.append(next_idx)
            else:
                mismatch_count += 1
                if mismatch_count > max_mismatches:
                    break
                cluster_indices.append(next_idx)
            j += 1
        
        # Filter cluster to dominant base activity
        if len(cluster_indices) > 0:
            cluster_activities = df.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.value_counts().idxmax()
            filtered_indices = [idx for idx in cluster_indices 
                              if df.at[idx, 'ProcessedActivity'] == dominant_activity]
            
            # Validate cluster
            if len(filtered_indices) >= min_matching_events:
                # Mark events for removal
                has_unsuffixed = False
                for idx in filtered_indices:
                    original_activity = df.at[idx, 'BaseActivity']
                    processed = preprocess_activity(original_activity)
                    if original_activity.lower() == processed.lower():
                        has_unsuffixed = True
                        break
                
                for idx in filtered_indices:
                    original_activity = df.at[idx, 'BaseActivity']
                    processed = preprocess_activity(original_activity)
                    df.at[idx, 'CollateralGroup'] = cluster_counter
                    
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
                
                cluster_counter += 1
        
        i = j

# Calculate detection metrics if label column exists
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        if isinstance(label, str) and 'collateral' in label.lower():
            return 1
        return 0
    
    y_true = df['label'].apply(normalize_label)
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
output_columns = ['Case', 'Timestamp', 'Activity']
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
if 'label' in df.columns:
    output_columns.append('label')

df_fixed['Activity_fixed'] = df_fixed['Activity']
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_fixed[output_columns].to_csv('data/pub/pub_collateral_cleaned_run3.csv', index=False)

# Summary statistics
original_count = len(df)
final_count = len(df_fixed)
removed_count = original_count - final_count
removed_pct = (removed_count / original_count) * 100

original_activities = df['Activity'].nunique()
final_activities = df_fixed['Activity'].nunique()

print("\n=== Summary Statistics ===")
print(f"Total events (original): {original_count}")
print(f"Total events (after removal): {final_count}")
print(f"Events removed: {removed_count} ({removed_pct:.2f}%)")
print(f"Unique activities before removal: {original_activities}")
print(f"Unique activities after removal: {final_activities}")
print(f"Output file path: data/pub/pub_collateral_cleaned_run3.csv")

# Print sample of removed events
removed_events = df[df['is_collateral_event'] == 1]
if not removed_events.empty:
    print("\nSample of removed events (up to 10):")
    sample = removed_events[['Case', 'Activity', 'Timestamp']].head(10)
    print(sample.to_string(index=False))

print(f"\nRun 3: Processed dataset saved to: data/pub/pub_collateral_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: collateral")