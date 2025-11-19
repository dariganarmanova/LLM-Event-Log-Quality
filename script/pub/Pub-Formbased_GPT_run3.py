# Generated script for Pub-Formbased - Run 3
# Generated on: 2025-11-14T13:28:47.955470
# Model: gpt-4o-2024-11-20

import pandas as pd
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

# File paths
input_file = 'data/pub/Pub-Formbased.csv'
output_file = 'data/pub/pub_form_based_cleaned_run3.csv'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = ['Case', 'Activity', 'Timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert Timestamp column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # Drop rows with invalid timestamps
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Sort by Case and Timestamp
    df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Step 2: Identify Flattened Events
    df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp']
    group_counts = df.groupby('group_key').size()
    df['is_flattened'] = df['group_key'].map(lambda x: 1 if group_counts[x] >= 2 else 0)

    # Step 3: Preprocess Flattened Groups
    normal_events = df[df['is_flattened'] == 0].copy()
    flattened_events = df[df['is_flattened'] == 1].copy()

    # Step 4: Merge Flattened Activities
    merged_flattened = (
        flattened_events.groupby(['Case', 'Timestamp'])
        .agg({
            'Activity': lambda x: ';'.join(sorted(x)),
            'label': 'first' if 'label' in df.columns else lambda x: None
        })
        .reset_index()
    )

    # Step 5: Combine and Sort
    final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
    final_df = final_df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Drop helper columns
    final_df = final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore')

    # Step 6: Calculate Detection Metrics
    if 'label' in df.columns:
        y_true = df['label'].notnull().astype(int)
        y_pred = df['is_flattened']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"{'✓' if precision >= 0.6 else '✗'} Precision threshold (≥ 0.6) met")
    else:
        print("=== Detection Performance Metrics ===")
        print("No labels available for metric calculation.")
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")

    # Step 7: Integrity Check
    total_flattened_groups = len(flattened_events['group_key'].unique())
    total_flattened_events = len(flattened_events)
    percentage_flattened = (total_flattened_events / len(df)) * 100
    print(f"Total flattened groups detected: {total_flattened_groups}")
    print(f"Total events marked as flattened: {total_flattened_events}")
    print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

    # Step 8: Save Output
    final_df.to_csv(output_file, index=False)

    # Step 9: Summary Statistics
    total_events = len(df)
    merged_events = len(merged_flattened)
    unique_activities_before = len(df['Activity'].unique())
    unique_activities_after = len(final_df['Activity'].unique())
    reduction_percentage = ((total_events - len(final_df)) / total_events) * 100

    print(f"Run 3: Total number of events: {total_events}")
    print(f"Run 3: Number of flattened (merged) events detected: {merged_events}")
    print(f"Run 3: Number of unique activities before merging: {unique_activities_before}")
    print(f"Run 3: Number of unique activities after merging: {unique_activities_after}")
    print(f"Run 3: Total reduction percentage: {reduction_percentage:.2f}%")
    print(f"Run 3: Processed dataset saved to: {output_file}")
    print(f"Run 3: Final dataset shape: {final_df.shape}")

except Exception as e:
    print(f"An error occurred during processing: {e}")