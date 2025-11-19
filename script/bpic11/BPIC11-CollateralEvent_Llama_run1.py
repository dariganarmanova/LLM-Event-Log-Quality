# Generated script for BPIC11-CollateralEvent - Run 1
# Generated on: 2025-11-13T11:39:55.195489
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
input_file = './data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'

label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")

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
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(activity_suffix_pattern, '')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('-', ' ')

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case].copy()
    case_df.sort_values(by=timestamp_column, inplace=True)
    cluster_start_time = None
    cluster_base_activity = None
    mismatch_count = 0
    cluster_events = []
    
    for i, event in case_df.iterrows():
        if cluster_start_time is None:
            cluster_start_time = event[timestamp_column]
            cluster_base_activity = event['ProcessedActivity']
            cluster_events.append(event)
            continue
        
        time_diff = (event[timestamp_column] - cluster_start_time).total_seconds()
        if time_diff > time_threshold:
            break
        
        if event['ProcessedActivity'] == cluster_base_activity:
            cluster_events.append(event)
        else:
            mismatch_count += 1
            if mismatch_count <= max_mismatches:
                cluster_events.append(event)
            else:
                break
    
    if len(cluster_events) >= min_matching_events:
        dominant_base = cluster_events[0]['ProcessedActivity']
        cluster_events = [event for event in cluster_events if event['ProcessedActivity'] == dominant_base]
        
        for event in cluster_events:
            if event['ProcessedActivity'] == dominant_base:
                df.loc[df.index[df[case_column] == case][i], 'is_collateral_event'] = 0
            else:
                df.loc[df.index[df[case_column] == case][i], 'is_collateral_event'] = 1
        
        df.loc[df[case_column] == case, 'CollateralGroup'] = cluster_counter
        cluster_counter += 1

# Mark Events for Removal
for cluster in df['CollateralGroup'].unique():
    cluster_df = df[df['CollateralGroup'] == cluster].copy()
    unsuffixed_event = cluster_df[cluster_df['ProcessedActivity'] == cluster_df['BaseActivity']].head(1)
    
    if not unsuffixed_event.empty:
        df.loc[df.index[df['CollateralGroup'] == cluster], 'is_collateral_event'] = 0
    else:
        df.loc[df.index[df['CollateralGroup'] == cluster].head(1), 'is_collateral_event'] = 0

# Calculate Detection Metrics
if label_column in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or label == '':
            return 0
        elif 'collateral' in label.lower():
            return 1
        else:
            return 0
    
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
else:
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Integrity Check
total_clusters = df['CollateralGroup'].nunique()
total_events_marked = df['is_collateral_event'].sum()
clean_events = (df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)
total_clean_events = clean_events.sum()

print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_events_marked}")
print(f"Events to be removed: {total_events_marked}")
print(f"Clean events that were NOT modified: {total_clean_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save Output
df_fixed['Activity_fixed'] = df_fixed['Activity']
if label_column in df.columns:
    df_fixed['label'] = df_fixed[label_column]
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total events (original): {df.shape[0]}")
print(f"Total events (after removal): {df_fixed.shape[0]}")
print(f"Events removed (count and percentage): {df.shape[0] - df_fixed.shape[0]} ({(df.shape[0] - df_fixed.shape[0]) / df.shape[0] * 100:.2f}%)")
print(f"Unique activities before removal: {df['Activity'].nunique()}")
print(f"Unique activities after removal: {df_fixed['Activity'].nunique()}")
print(f"Output file path: {output_file}")

# Print sample of up to 10 removed events
removed_events = df[(df['is_collateral_event'] == 1) & (df['iscollateral'] == 0)].head(10)
print(removed_events)