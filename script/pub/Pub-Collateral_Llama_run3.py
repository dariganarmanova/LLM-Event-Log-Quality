# Generated script for Pub-Collateral - Run 3
# Generated on: 2025-11-14T13:25:31.674358
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

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp', 'Variant']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Convert Timestamp to datetime format using robust parsing
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')

# Sort by Case and Timestamp
df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Identify Collateral Activities
collateral_suffix = ':collateral'
df['iscollateral'] = df['Activity'].str.endswith(collateral_suffix)
df['BaseActivity'] = df['Activity'].str.replace(collateral_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(activity_suffix_pattern, '', x.lower()))
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('_', ' ').str.replace('-', ' ').str.strip()

# Initialize Tracking Columns
df['CollateralGroup'] = -1
df['is_collateral_event'] = 0
cluster_counter = 0

# Sliding Window Clustering
for case in df['Case'].unique():
    case_df = df[df['Case'] == case]
    case_df.sort_values(by='Timestamp', inplace=True)
    
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
        
        # Filter cluster to dominant base activity
        base_counts = case_df[case_df['Case'] == case]['ProcessedActivity'].value_counts()
        dominant_base = base_counts.idxmax()
        
        # Validate cluster
        if len(cluster) >= min_matching_events:
            # Mark events for removal
            for event in cluster:
                if event['ProcessedActivity'] == dominant_base:
                    df.loc[df['Case'] == case and df['ProcessedActivity'] == dominant_base, 'is_collateral_event'] = 0
                else:
                    df.loc[df['Case'] == case and df['ProcessedActivity'] != dominant_base, 'is_collateral_event'] = 1
            
            # Assign cluster ID
            df.loc[df['Case'] == case, 'CollateralGroup'] = cluster_counter
            
            # Increment cluster counter
            cluster_counter += 1

# Calculate Detection Metrics
label_column = 'label'
if label_column in df.columns:
    def normalize_label(x):
        if pd.isnull(x) or pd.isna(x):
            return 0
        elif 'collateral' in x.lower():
            return 1
        else:
            return 0
    
    df['y_true'] = df[label_column].apply(normalize_label)
    y_pred = df['is_collateral_event']
    precision = precision_score(y_true=df['y_true'], y_pred=y_pred, zero_division=0)
    recall = recall_score(y_true=df['y_true'], y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=df['y_true'], y_pred=y_pred, zero_division=0)
    
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")

# Integrity Check
total_collateral_clusters = df['CollateralGroup'].nunique()
events_marked_collateral = df[df['is_collateral_event'] == 1].shape[0]
clean_events = df[(df['iscollateral'] == 0) & (df['is_collateral_event'] == 0)].shape[0]
print(f"Total collateral clusters detected: {total_collateral_clusters}")
print(f"Total events marked as collateral: {events_marked_collateral}")
print(f"Events to be removed: {events_marked_collateral}")

# Remove Collateral Events
df_fixed = df[df['is_collateral_event'] == 0]

# Save Output
output_file = 'data/pub/pub_collateral_cleaned_run3.csv'
df_fixed[['Case', 'Timestamp', 'Variant', 'Activity']].to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: collateral")
print(f"Run 3: Removed {events_marked_collateral} events ({(events_marked_collateral / df.shape[0]) * 100:.2f}%)")
print(f"Run 3: Unique activities before removal: {df['Activity'].nunique()}")
print(f"Run 3: Unique activities after removal: {df_fixed['Activity'].nunique()}")

# Print sample of up to 10 removed events
removed_events = df[df['is_collateral_event'] == 1].head(10)[['Case', 'Activity', 'Timestamp']]
print(f"Removed events:")
print(removed_events)