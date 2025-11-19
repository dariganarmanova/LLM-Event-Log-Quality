# Generated script for BPIC11-CollateralEvent - Run 3
# Generated on: 2025-11-13T11:38:52.116032
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0  # seconds
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
collateral_suffix = ':collateral'

# Input and output file paths
input_file = 'data/bpic11/BPIC11-CollateralEvent.csv'
output_file = 'data/bpic11/bpic11_collateral_cleaned_run3.csv'

# Columns
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    print("Error: Required columns are missing from the dataset.")
    exit()

# Convert Timestamp to datetime
try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
except Exception as e:
    print(f"Error converting timestamps: {e}")
    exit()

# Drop rows with invalid timestamps
df = df.dropna(subset=[timestamp_column])

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify collateral activities
df['iscollateral'] = df[activity_column].str.endswith(collateral_suffix).astype(int)

# Preprocess activity names
def preprocess_activity(activity):
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[_\-\s]+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Initialize tracking columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding window clustering
for case_id, case_events in df.groupby(case_column):
    case_events = case_events.reset_index()
    n = len(case_events)
    i = 0
    while i < n:
        cluster_start_time = case_events.loc[i, timestamp_column]
        base_activity = case_events.loc[i, 'ProcessedActivity']
        cluster_indices = [i]
        mismatch_count = 0

        for j in range(i + 1, n):
            time_diff = (case_events.loc[j, timestamp_column] - cluster_start_time).total_seconds()
            if time_diff > time_threshold:
                break

            current_activity = case_events.loc[j, 'ProcessedActivity']
            if current_activity == base_activity:
                cluster_indices.append(j)
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster_indices.append(j)
                else:
                    break

        # Validate cluster
        if len(cluster_indices) >= min_matching_events:
            cluster_activities = case_events.loc[cluster_indices, 'ProcessedActivity']
            dominant_activity = cluster_activities.mode()[0]
            valid_indices = case_events.index[cluster_indices][
                case_events.loc[cluster_indices, 'ProcessedActivity'] == dominant_activity
            ]

            # Mark events for removal
            unsuffixed_events = valid_indices[
                case_events.loc[valid_indices, activity_column] == dominant_activity
            ]
            if not unsuffixed_events.empty:
                keep_index = unsuffixed_events[0]
            else:
                keep_index = valid_indices[0]

            case_events.loc[valid_indices, 'is_collateral_event'] = 1
            case_events.loc[keep_index, 'is_collateral_event'] = 0
            case_events.loc[valid_indices, 'CollateralGroup'] = cluster_counter
            cluster_counter += 1

        i = cluster_indices[-1] + 1

    # Update the main DataFrame
    df.loc[case_events.index, 'is_collateral_event'] = case_events['is_collateral_event']
    df.loc[case_events.index, 'CollateralGroup'] = case_events['CollateralGroup']

# Calculate detection metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].fillna('').str.contains('collateral', case=not case_sensitive).astype(int)
    y_true = df['y_true']
    y_pred = df['is_collateral_event']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation.")
    precision = recall = f1 = 0.0

# Integrity check
total_clusters = df['CollateralGroup'].nunique() - (1 if -1 in df['CollateralGroup'].unique() else 0)
total_collateral_events = df['is_collateral_event'].sum()
total_clean_events = len(df) - total_collateral_events
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {total_collateral_events}")
print(f"Events to be removed: {total_collateral_events}")

# Remove collateral events
df_fixed = df[df['is_collateral_event'] == 0].copy()

# Save output
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Total events (original): {len(df)}")
print(f"Run 3: Total events (after removal): {len(df_fixed)}")
print(f"Run 3: Events removed: {total_collateral_events} ({(total_collateral_events / len(df)) * 100:.2f}%)")
print(f"Run 3: Unique activities before removal: {df[activity_column].nunique()}")
print(f"Run 3: Unique activities after removal: {df_fixed[activity_column].nunique()}")