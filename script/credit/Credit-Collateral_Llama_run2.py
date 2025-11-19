# Generated script for Credit-Collateral - Run 2
# Generated on: 2025-11-13T16:03:10.469409
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
output_file = 'data/credit/credit_collateral_cleaned_run2.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp', 'Variant']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Convert Timestamp to datetime format using robust parsing
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# Sort by Case and Timestamp
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify Collateral Activities
df['iscollateral'] = df['Activity'].str.endswith(':collateral')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(activity_suffix_pattern, '', x).lower().replace('_', ' ').replace('-', ' '))

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
        
        # Expand cluster window
        for j in range(i + 1, len(case_df)):
            time_diff = (case_df.iloc[j]['Timestamp'] - cluster_start_time).total_seconds()
            if time_diff > time_threshold or mismatch_count > max_mismatches:
                break
            if case_df.iloc[j]['ProcessedActivity'] == base_activity:
                cluster.append(case_df.iloc[j])
            else:
                mismatch_count += 1
        
        # Filter cluster to dominant base activity
        base_counts = case_df['ProcessedActivity'].value_counts()
        dominant_base = base_counts.idxmax()
        cluster = [event for event in cluster if event['ProcessedActivity'] == dominant_base]
        
        # Validate cluster
        if len(cluster) >= min_matching_events:
            # Mark events for removal
            for event in cluster:
                if event['Activity'] == event['ProcessedActivity']:
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
        elif 'collateral' in x.lower():
            return 1
        else:
            return 0
    
    y_true = df['label'].apply(normalize_label)
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
    print("No labels available for metric calculation")

# Integrity Check
total_collateral_clusters = df['CollateralGroup'].nunique()
events_marked_collateral = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[df['iscollateral'] == 0 and df['is_collateral_event'] == 0].shape[0]
print(f"Total collateral clusters detected: {total_collateral_clusters}")
print(f"Total events marked as collateral: {events_marked_collateral}")
print(f"Events to be removed: {events_marked_collateral}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0]

# Save Output
df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']].to_csv(output_file, index=False)

# Summary Statistics
print(f"Total events (original): {df.shape[0]}")
print(f"Total events (after removal): {df_fixed.shape[0]}")
print(f"Events removed (count and percentage): {events_marked_collateral} ({(events_marked_collateral / df.shape[0]) * 100:.2f}%)")
print(f"Unique activities before removal: {df['Activity'].nunique()}")
print(f"Unique activities after removal: {df_fixed['Activity'].nunique()}")
print(f"Output file path: {output_file}")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(removed_events)