# Generated script for BPIC15-CollateralEvent - Run 3
# Generated on: 2025-11-13T14:49:20.528079
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
import os

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
output_file = 'data/bpic15/bpic15_collateral_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
collateral_suffix = ':collateral'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns not found in the input CSV")

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
df['ProcessedActivity'] = df['BaseActivity'].str.lower().str.replace(r'[_\-\s]+', ' ').str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df[case_column].unique():
    case_df = df[df[case_column] == case].copy()
    cluster_start_time = case_df[timestamp_column].iloc[0]
    base_activity = case_df['ProcessedActivity'].iloc[0]
    mismatch_count = 0
    for i in range(len(case_df)):
        if i > 0:
            time_diff = (case_df[timestamp_column].iloc[i] - cluster_start_time).total_seconds()
            if time_diff > time_threshold or mismatch_count > max_mismatches:
                break
            if case_df['ProcessedActivity'].iloc[i] == base_activity:
                case_df['CollateralGroup'].iloc[i] = cluster_counter
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    case_df['CollateralGroup'].iloc[i] = cluster_counter
        cluster_start_time = case_df[timestamp_column].iloc[i]
        base_activity = case_df['ProcessedActivity'].iloc[i]
    # Filter cluster to dominant base activity
    cluster_counts = case_df['ProcessedActivity'].value_counts()
    dominant_base = cluster_counts.index[0]
    case_df.loc[case_df['ProcessedActivity'] != dominant_base, 'CollateralGroup'] = -1
    # Validate cluster
    if len(case_df[case_df['CollateralGroup'] != -1]) >= min_matching_events:
        cluster_counter += 1
        # Mark events for removal
        unsuffixed_event = case_df[case_df['CollateralGroup'] == cluster_counter][activity_column].str.replace(activity_suffix_pattern, '').str.lower().str.contains(base_activity).any()
        if unsuffixed_event:
            case_df.loc[case_df['CollateralGroup'] == cluster_counter, 'is_collateral_event'] = 0
        else:
            case_df.loc[case_df['CollateralGroup'] == cluster_counter, 'is_collateral_event'] = 1

# Calculate Detection Metrics
if label_column in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or label == 'NaN':
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
    if precision >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("No labels available for metric calculation")
    precision = 0.0
    recall = 0.0
    f1 = 0.0

# Integrity Check
total_collateral_clusters = df['CollateralGroup'].nunique() - 1
total_events_marked = len(df[df['is_collateral_event'] == 1])
clean_events = len(df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)])
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
print(f"Total events (original): {len(df)}")
print(f"Total events (after removal): {len(df_fixed)}")
print(f"Events removed (count and percentage): {total_events_marked} ({(total_events_marked / len(df)) * 100:.2f}%)")
print(f"Unique activities before removal: {len(df['Activity'].unique())}")
print(f"Unique activities after removal: {len(df_fixed['Activity'].unique())}")
print(f"Output file path: {output_file}")
print(f"Sample of up to 10 removed events:")
print(df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']])

print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: collateral")