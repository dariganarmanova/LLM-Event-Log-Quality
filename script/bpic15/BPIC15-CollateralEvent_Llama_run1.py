# Generated script for BPIC15-CollateralEvent - Run 1
# Generated on: 2025-11-13T14:49:11.877897
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
input_file = 'data/bpic15/BPIC15-CollateralEvent.csv'
output_file = 'data/bpic15/bpic15_collateral_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(column in df.columns for column in required_columns):
    raise ValueError("Required columns are missing from the input data")

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
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(' ', ' ')

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case]
    cluster_start_time = case_df.iloc[0][timestamp_column]
    base_activity = case_df.iloc[0]['ProcessedActivity']
    cluster = [case_df.iloc[0].index]
    mismatch_count = 0
    for i in range(1, len(case_df)):
        event_time = case_df.iloc[i][timestamp_column]
        time_diff = (event_time - cluster_start_time).total_seconds()
        if time_diff > time_threshold:
            break
        event_activity = case_df.iloc[i]['ProcessedActivity']
        if event_activity == base_activity:
            cluster.append(case_df.iloc[i].index)
        else:
            mismatch_count += 1
            if mismatch_count <= max_mismatches:
                cluster.append(case_df.iloc[i].index)
            else:
                break
    if len(cluster) >= min_matching_events:
        dominant_base = case_df.loc[cluster, 'ProcessedActivity'].value_counts().index[0]
        cluster_df = case_df.loc[cluster]
        cluster_df['is_collateral_event'] = 0
        cluster_df.loc[cluster_df['ProcessedActivity'] != dominant_base, 'is_collateral_event'] = 1
        cluster_df['CollateralGroup'] = cluster_counter
        cluster_counter += 1
        for idx in cluster:
            df.at[idx, 'CollateralGroup'] = cluster_counter - 1
            df.at[idx, 'is_collateral_event'] = cluster_df.loc[idx, 'is_collateral_event']

# Mark Events for Removal
for group in df['CollateralGroup'].unique():
    group_df = df[df['CollateralGroup'] == group]
    unsuffixed_event = group_df[group_df['ProcessedActivity'] == group_df['BaseActivity']].index[0]
    group_df.loc[group_df.index != unsuffixed_event, 'is_collateral_event'] = 1

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
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Integrity Check
total_clusters = len(df['CollateralGroup'].unique())
total_collateral = len(df[df['is_collateral_event'] == 1])
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_collateral}")
print(f"Events to be removed: {total_collateral}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0]

# Save Output
df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']] = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']].copy()
df_fixed['Activity_fixed'] = df_fixed['Activity']
if label_column in df.columns:
    df_fixed[label_column] = df_fixed[label_column].copy()
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total events (original): {len(df)}")
print(f"Total events (after removal): {len(df_fixed)}")
print(f"Events removed: {len(df) - len(df_fixed)} ({(len(df) - len(df_fixed)) / len(df) * 100:.2f}%)")
print(f"Unique activities before removal: {len(df['Activity'].unique())}")
print(f"Unique activities after removal: {len(df_fixed['Activity'].unique())}")
print(f"Output file path: {output_file}")
print(f"Sample of removed events:")
print(df[df['is_collateral_event'] == 1].head(10))