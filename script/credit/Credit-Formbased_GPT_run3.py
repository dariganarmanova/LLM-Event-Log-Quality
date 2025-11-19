# Generated script for Credit-Formbased - Run 3
# Generated on: 2025-11-13T16:18:53.662972
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

# Input and output file paths
input_file = 'data/credit/Credit-Formbased.csv'
output_file = 'data/credit/credit_form_based_cleaned_run3.csv'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = ['Case', 'Activity', 'Timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Rename common variants if necessary
    df.rename(columns={'CaseID': 'Case', 'ActivityID': 'Activity', 'Time': 'Timestamp'}, inplace=True)

    # Convert Timestamp to datetime and standardize format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    if df['Timestamp'].isnull().any():
        raise ValueError("Invalid timestamps detected. Ensure all timestamps are in a valid format.")

    # Sort by Case and Timestamp
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

    # Step 2: Identify Flattened Events
    df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp'].astype(str)
    group_counts = df.groupby('group_key')['Activity'].transform('count')
    df['is_flattened'] = (group_counts >= 2).astype(int)

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
    final_df.sort_values(by=['Case', 'Timestamp'], inplace=True)
    final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore', inplace=True)

    # Step 6: Calculate Detection Metrics (if label column exists)
    if 'label' in df.columns:
        y_true = df['label'].notna().astype(int)
        y_pred = df['is_flattened']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print("\n=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"{'✓' if precision >= 0.6 else '✗'} Precision threshold (≥ 0.6) met")
    else:
        print("\n=== Detection Performance Metrics ===")
        print("No labels available for metric calculation.")
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")

    # Step 7: Integrity Check
    total_flattened_groups = len(flattened_events['group_key'].unique())
    total_flattened_events = len(flattened_events)
    percentage_flattened = (total_flattened_events / len(df)) * 100

    print("\n=== Integrity Check ===")
    print(f"Total flattened groups detected: {total_flattened_groups}")
    print(f"Total events marked as flattened: {total_flattened_events}")
    print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

    # Step 8: Save Output
    final_df.to_csv(output_file, index=False)

    # Step 9: Summary Statistics
    total_events = len(df)
    total_merged_events = len(merged_flattened)
    unique_activities_before = df['Activity'].nunique()
    unique_activities_after = final_df['Activity'].nunique()
    reduction_percentage = ((total_events - len(final_df)) / total_events) * 100

    print("\n=== Summary Statistics ===")
    print(f"Total number of events: {total_events}")
    print(f"Number of flattened (merged) events detected: {total_merged_events}")
    print(f"Number of unique activities before merging: {unique_activities_before}")
    print(f"Number of unique activities after merging: {unique_activities_after}")
    print(f"Total reduction percentage: {reduction_percentage:.2f}%")
    print(f"Output file path: {output_file}")
    print("\nSample of merged activities:")
    print(merged_flattened.head(10))

except Exception as e:
    print(f"An error occurred: {e}")