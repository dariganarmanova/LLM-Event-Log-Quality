# Generated script for BPIC15-CollateralEvent - Run 2
# Generated on: 2025-11-13T14:49:15.722962
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

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
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns missing from the dataset")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Convert Timestamp to datetime format using robust parsing
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="mixed")

# Sort by Case and Timestamp
df.sort_values(by=[case_column, timestamp_column], inplace=True)

# Identify Collateral Activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix)
df['BaseActivity'] = df[activity_column].str.replace(activity_suffix_pattern, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_\d+', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_signed\d*', ' signed')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case].copy()
    case_df.sort_values(by=timestamp_column, inplace=True)
    cluster_start_time = case_df.iloc[0][timestamp_column]
    base_activity = case_df.iloc[0]['ProcessedActivity']
    mismatch_count = 0
    for i in range(1, len(case_df)):
        event_timestamp = case_df.iloc[i][timestamp_column]
        time_diff = (event_timestamp - cluster_start_time).total_seconds()
        if time_diff > time_threshold or mismatch_count > max_mismatches:
            break
        if case_df.iloc[i]['ProcessedActivity'] == base_activity:
            case_df.at[case_df.index[i], 'CollateralGroup'] = cluster_counter
            case_df.at[case_df.index[i], 'is_collateral_event'] = 0
        else:
            mismatch_count += 1
            if mismatch_count <= max_mismatches:
                case_df.at[case_df.index[i], 'CollateralGroup'] = cluster_counter
                case_df.at[case_df.index[i], 'is_collateral_event'] = 0
        cluster_start_time = event_timestamp
    # Filter cluster to dominant base activity
    cluster_df = case_df[case_df['CollateralGroup'] == cluster_counter]
    base_counts = cluster_df['ProcessedActivity'].value_counts()
    dominant_base = base_counts.index[0]
    cluster_df = cluster_df[cluster_df['ProcessedActivity'] == dominant_base]
    # Mark events for removal
    if len(cluster_df) >= min_matching_events:
        unsuffixed_event = cluster_df[cluster_df['BaseActivity'] == cluster_df['ProcessedActivity']]
        if not unsuffixed_event.empty:
            unsuffixed_event['is_collateral_event'] = 0
            suffixed_events = cluster_df[cluster_df['BaseActivity'] != cluster_df['ProcessedActivity']]
            suffixed_events['is_collateral_event'] = 1
            cluster_df.update(unsuffixed_event)
            cluster_df.update(suffixed_events)
        else:
            cluster_df['is_collateral_event'] = cluster_df.index // 2
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

# Integrity Check
total_clusters = df['CollateralGroup'].nunique()
collateral_events = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {collateral_events}")
print(f"Events to be removed: {collateral_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save Output
df_fixed['Activity_fixed'] = df_fixed['Activity']
if label_column in df_fixed.columns:
    df_fixed = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity_fixed', label_column]]
else:
    df_fixed = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity_fixed']]
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 2: Total events (original): {df.shape[0]}")
print(f"Run 2: Total events (after removal): {df_fixed.shape[0]}")
print(f"Run 2: Events removed: {df.shape[0] - df_fixed.shape[0]} ({(df.shape[0] - df_fixed.shape[0]) / df.shape[0] * 100:.2f}%)")
print(f"Run 2: Unique activities before removal: {df['Activity'].nunique()}")
print(f"Run 2: Unique activities after removal: {df_fixed['Activity_fixed'].nunique()}")
print(f"Run 2: Output file path: {output_file}")
print(f"Run 2: Sample of removed events:")
print(df[df['is_collateral_event'] == 1].head(10))