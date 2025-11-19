# Generated script for BPIC11-CollateralEvent - Run 2
# Generated on: 2025-11-13T11:39:58.321173
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
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing from the input file.")

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
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_\d+', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ')

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case]
    for i in range(len(case_df)):
        cluster = [case_df.iloc[i]]
        cluster_start_time = case_df.iloc[i][timestamp_column]
        base_activity = case_df.iloc[i]['ProcessedActivity']
        mismatch_count = 0
        while i + 1 < len(case_df) and (case_df.iloc[i + 1][timestamp_column] - cluster_start_time).total_seconds() <= time_threshold:
            if case_df.iloc[i + 1]['ProcessedActivity'] == base_activity:
                cluster.append(case_df.iloc[i + 1])
                i += 1
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster.append(case_df.iloc[i + 1])
                    i += 1
                else:
                    break
        # Filter cluster to dominant base activity
        base_counts = case_df['ProcessedActivity'].value_counts()
        dominant_base = base_counts.idxmax()
        cluster = [event for event in cluster if event['ProcessedActivity'] == dominant_base]
        # Validate cluster
        if len(cluster) >= min_matching_events:
            # Mark events for removal
            for event in cluster:
                if event['Activity'] == event['BaseActivity']:
                    df.loc[df['Case'] == case and df['Timestamp'] == event['Timestamp'], 'is_collateral_event'] = 0
                else:
                    df.loc[df['Case'] == case and df['Timestamp'] == event['Timestamp'], 'is_collateral_event'] = 1
            # Assign cluster ID
            df.loc[df['Case'] == case, 'CollateralGroup'] = cluster_counter
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
total_collateral_clusters = df['CollateralGroup'].nunique()
total_events_marked = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_collateral_clusters}")
print(f"Total events marked as collateral: {total_events_marked}")
print(f"Events to be removed: {total_events_marked}")

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
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(removed_events)

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: collateral")