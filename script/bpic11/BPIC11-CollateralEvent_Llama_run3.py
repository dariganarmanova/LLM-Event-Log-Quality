# Generated script for BPIC11-CollateralEvent - Run 3
# Generated on: 2025-11-13T11:40:00.532984
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
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns not found in the input file")

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
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_\d+', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_', ' ')

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case].copy()
    case_df.sort_values(by=timestamp_column, inplace=True)
    for i in range(len(case_df)):
        cluster_start_time = case_df.iloc[i][timestamp_column]
        base_activity = case_df.iloc[i]['ProcessedActivity']
        mismatch_count = 0
        cluster = [i]
        for j in range(i + 1, len(case_df)):
            time_diff = (case_df.iloc[j][timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break
            if case_df.iloc[j]['ProcessedActivity'] == base_activity:
                cluster.append(j)
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster.append(j)
                else:
                    break
        if len(cluster) >= min_matching_events:
            dominant_base = case_df.iloc[cluster[0]]['ProcessedActivity']
            cluster_df = case_df.iloc[cluster].copy()
            cluster_df['ProcessedActivity'] = cluster_df['ProcessedActivity'].apply(lambda x: x == dominant_base)
            cluster_df['is_collateral_event'] = 0
            cluster_df.loc[cluster_df['ProcessedActivity'] == False, 'is_collateral_event'] = 1
            cluster_df['CollateralGroup'] = cluster_counter
            for idx in cluster_df.index:
                df.loc[idx, 'CollateralGroup'] = cluster_counter
                df.loc[idx, 'is_collateral_event'] = cluster_df.loc[idx, 'is_collateral_event']
            cluster_counter += 1

# Mark Events for Removal
for group in df['CollateralGroup'].unique():
    group_df = df[df['CollateralGroup'] == group].copy()
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
total_collateral_events = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_collateral_events}")
print(f"Events to be removed: {total_collateral_events}")
print(f"Clean events that were NOT modified: {clean_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save Output
df_fixed['Activity_fixed'] = df_fixed['Activity']
if label_column in df.columns:
    df_fixed = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity_fixed', label_column]]
else:
    df_fixed = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity_fixed']]
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
total_events_original = df.shape[0]
total_events_after_removal = df_fixed.shape[0]
events_removed = total_events_original - total_events_after_removal
percentage_removed = (events_removed / total_events_original) * 100
unique_activities_before_removal = df['Activity'].nunique()
unique_activities_after_removal = df_fixed['Activity_fixed'].nunique()
print(f"Total events (original): {total_events_original}")
print(f"Total events (after removal): {total_events_after_removal}")
print(f"Events removed (count and percentage): {events_removed} ({percentage_removed:.2f}%)")
print(f"Unique activities before removal: {unique_activities_before_removal}")
print(f"Unique activities after removal: {unique_activities_after_removal}")
print(f"Output file path: {output_file}")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(removed_events)