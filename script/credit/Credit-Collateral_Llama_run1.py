# Generated script for Credit-Collateral - Run 1
# Generated on: 2025-11-13T16:03:07.455489
# Model: meta-llama/Llama-3.1-8B-Instruct

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
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, label_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing from the input file")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Convert Timestamp to datetime format using robust parsing
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by Case and Timestamp
df.sort_values(by=[case_column, timestamp_column], inplace=True)

# Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix)
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].replace(to_replace=activity_suffix_pattern, value='', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('-', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case]
    cluster_start_time = case_df.iloc[0][timestamp_column]
    cluster_base_activity = case_df.iloc[0]['ProcessedActivity']
    cluster = [case_df.iloc[0]]
    mismatch_count = 0
    
    for i in range(1, len(case_df)):
        event = case_df.iloc[i]
        time_diff = (event[timestamp_column] - cluster_start_time).total_seconds()
        
        if time_diff > time_threshold:
            break
        
        if event['ProcessedActivity'] == cluster_base_activity:
            cluster.append(event)
        else:
            mismatch_count += 1
            
            if mismatch_count <= max_mismatches:
                cluster.append(event)
            else:
                break
    
    # Filter cluster to dominant base activity
    base_counts = case_df['ProcessedActivity'].value_counts()
    dominant_base = base_counts.index[0]
    
    if len(cluster) >= min_matching_events:
        # Mark events for removal
        for event in cluster:
            if event['ProcessedActivity'] == dominant_base:
                df.loc[df['Case'] == event['Case'] and df['Timestamp'] == event['Timestamp'], 'is_collateral_event'] = 0
            else:
                df.loc[df['Case'] == event['Case'] and df['Timestamp'] == event['Timestamp'], 'is_collateral_event'] = 1
        
        # Assign cluster ID
        for event in cluster:
            df.loc[df['Case'] == event['Case'] and df['Timestamp'] == event['Timestamp'], 'CollateralGroup'] = cluster_counter
        cluster_counter += 1

# Calculate Detection Metrics
if label_column in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or label == '':
            return 0
        elif 'collateral' in label.lower():
            return 1
        else:
            return 0
    
    df['y_true'] = df[label_column].apply(normalize_label)
    df['y_pred'] = df['is_collateral_event']
    
    precision = precision_score(df['y_true'], df['y_pred'], zero_division=0)
    recall = recall_score(df['y_true'], df['y_pred'], zero_division=0)
    f1 = f1_score(df['y_true'], df['y_pred'], zero_division=0)
    
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
else:
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Integrity Check
total_collateral_clusters = df['CollateralGroup'].nunique()
total_collateral_events = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_collateral_clusters}")
print(f"Total events marked as collateral: {total_collateral_events}")
print(f"Events to be removed: {total_collateral_events}")
print(f"Clean events that were NOT modified: {clean_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save Output
df_fixed[['Case', 'Timestamp', 'Variant', activity_column]] = df_fixed[['Case', 'Timestamp', 'Variant', activity_column]].astype(str)
df_fixed[timestamp_column] = df_fixed[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
if label_column in df.columns:
    df_fixed[label_column] = df_fixed[label_column].astype(str)
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total events (original): {df.shape[0]}")
print(f"Total events (after removal): {df_fixed.shape[0]}")
print(f"Events removed (count and percentage): {total_collateral_events} ({(total_collateral_events / df.shape[0]) * 100:.2f}%)")
print(f"Unique activities before removal: {df[activity_column].nunique()}")
print(f"Unique activities after removal: {df_fixed[activity_column].nunique()}")
print(f"Output file path: {output_file}")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', activity_column, timestamp_column]]
print(f"Removed events (sample of up to 10):")
print(removed_events)