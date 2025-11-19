# Generated script for Credit-Collateral - Run 3
# Generated on: 2025-11-13T16:03:12.377048
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# Load the data
input_file = 'data/credit/Credit-Collateral.csv'
output_file = 'data/credit/credit_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, label_column]
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
df['BaseActivity'] = df[activity_column].str.replace(collateral_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r"(_signed\d*|_\d+)$", '')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r"[_\-]", ' ')

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
        for j in range(i + 1, len(case_df)):
            time_diff = (case_df.iloc[j][timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold or mismatch_count > max_mismatches:
                break
            if case_df.iloc[j]['ProcessedActivity'] == base_activity:
                cluster.append(case_df.iloc[j])
            else:
                mismatch_count += 1
        # Filter cluster to dominant base activity
        base_counts = case_df[case_df['ProcessedActivity'] == base_activity].shape[0]
        dominant_base = case_df['ProcessedActivity'].value_counts().index[0]
        cluster = cluster[[x['ProcessedActivity'] == dominant_base for x in cluster]]
        # Validate cluster
        if cluster.shape[0] >= min_matching_events:
            # Mark events for removal
            if any(x['ProcessedActivity'] == x['BaseActivity'] for x in cluster):
                keep_event = cluster[cluster['ProcessedActivity'] == cluster['BaseActivity']].index[0]
                df.loc[keep_event, 'is_collateral_event'] = 0
                for index in cluster.index:
                    if index != keep_event:
                        df.loc[index, 'is_collateral_event'] = 1
            else:
                first_event = cluster.index[0]
                df.loc[first_event, 'is_collateral_event'] = 0
                for index in cluster.index:
                    if index != first_event:
                        df.loc[index, 'is_collateral_event'] = 1
            # Assign cluster ID
            df.loc[cluster.index, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

# Calculate Detection Metrics
if label_column in df.columns:
    def normalize_label(x):
        if pd.isnull(x) or pd.isna(x):
            return 0
        elif 'collateral' in str(x).lower():
            return 1
        else:
            return 0

    df['y_true'] = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    precision = precision_score(df['y_true'], y_pred, zero_division=0)
    recall = recall_score(df['y_true'], y_pred, zero_division=0)
    f1 = f1_score(df['y_true'], y_pred, zero_division=0)
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
    print(f"Note: No labels available for metric calculation")

# Integrity Check
total_collateral_clusters = df['CollateralGroup'].nunique()
total_events_marked = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_collateral_clusters}")
print(f"Total events marked as collateral: {total_events_marked}")
print(f"Events to be removed: {total_events_marked}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0]

# Save Output
df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']] = df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']].applymap(lambda x: str(x))
if label_column in df.columns:
    df_fixed[label_column] = df_fixed[label_column].apply(lambda x: str(x))
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: collateral")
print(f"Run 3: Total events (original): {df.shape[0]}")
print(f"Run 3: Total events (after removal): {df_fixed.shape[0]}")
print(f"Run 3: Events removed (count and percentage): {total_events_marked} ({(total_events_marked / df.shape[0]) * 100:.2f}%)")
print(f"Run 3: Unique activities before removal: {df['Activity'].nunique()}")
print(f"Run 3: Unique activities after removal: {df_fixed['Activity'].nunique()}")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(f"Run 3: Removed events:")
print(removed_events)