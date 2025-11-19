# Generated script for Pub-Collateral - Run 1
# Generated on: 2025-11-14T13:25:27.249142
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
input_file = 'data/pub/Pub-Collateral.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist: Case, Activity, Timestamp, Variant
required_columns = ['Case', 'Activity', 'Timestamp', 'Variant']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing from the input CSV")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Convert Timestamp to datetime format using robust parsing
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# Sort by Case and Timestamp
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify Collateral Activities
df['iscollateral'] = df['Activity'].str.endswith(':collateral')
df['BaseActivity'] = df['Activity'].str.replace(':collateral', '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(activity_suffix_pattern, '', x).lower().replace('_', ' ').replace('-', ' ').replace(' ', ' ').strip())

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df['Case'].unique():
    case_df = df[df['Case'] == case]
    for i in range(len(case_df)):
        cluster = [case_df.iloc[i]]
        cluster_start_time = case_df.iloc[i]['Timestamp']
        base_activity = case_df.iloc[i]['ProcessedActivity']
        mismatch_count = 0
        for j in range(i + 1, len(case_df)):
            time_diff = (case_df.iloc[j]['Timestamp'] - cluster_start_time).total_seconds()
            if time_diff > time_threshold or mismatch_count > max_mismatches:
                break
            if case_df.iloc[j]['ProcessedActivity'] == base_activity:
                cluster.append(case_df.iloc[j])
            else:
                mismatch_count += 1
                if mismatch_count <= max_mismatches:
                    cluster.append(case_df.iloc[j])
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
if 'label' in df.columns:
    def normalize_label(x):
        if pd.isnull(x) or pd.isna(x):
            return 0
        elif 'collateral' in str(x).lower():
            return 1
        else:
            return 0
    df['y_true'] = df['label'].apply(normalize_label)
    y_true = df['y_true'].tolist()
    y_pred = df['is_collateral_event'].tolist()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
total_clusters = df['CollateralGroup'].nunique()
collateral_events = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_clusters}")
print(f"Total events marked as collateral: {collateral_events}")
print(f"Events to be removed: {collateral_events}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0]

# Save Output
df_fixed['Activity_fixed'] = df_fixed['Activity']
df_fixed['Timestamp'] = df_fixed['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
if 'label' in df_fixed.columns:
    df_fixed.to_csv('data/pub/pub_collateral_cleaned_run1.csv', index=False)
else:
    df_fixed[['Case', 'Timestamp', 'Variant', 'Activity_fixed']].to_csv('data/pub/pub_collateral_cleaned_run1.csv', index=False)

# Summary Statistics
print(f"Run 1: Total events (original): {df.shape[0]}")
print(f"Run 1: Total events (after removal): {df_fixed.shape[0]}")
print(f"Run 1: Events removed (count and percentage): {collateral_events} ({(collateral_events / df.shape[0]) * 100:.2f}%)")
print(f"Run 1: Unique activities before removal: {df['ProcessedActivity'].nunique()}")
print(f"Run 1: Unique activities after removal: {df_fixed['ProcessedActivity'].nunique()}")
print(f"Run 1: Output file path: data/pub/pub_collateral_cleaned_run1.csv")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(f"Run 1: Sample of removed events:")
print(removed_events)

print(f"Run 1: Processed dataset saved to: data/pub/pub_collateral_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: collateral")