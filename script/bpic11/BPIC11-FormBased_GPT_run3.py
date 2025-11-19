# Generated script for BPIC11-FormBased - Run 3
# Generated on: 2025-11-13T11:17:17.079724
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

# Input and output paths
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run3.csv'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = ['Case', 'Activity', 'Timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert Timestamp to datetime and standardize format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # Drop rows with invalid timestamps
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Sort by Case and Timestamp
    df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Step 2: Identify Flattened Events
    df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp']
    group_counts = df.groupby('group_key')['Activity'].transform('count')
    df['is_flattened'] = (group_counts >= 2).astype(int)

    # Step 3: Preprocess Flattened Groups
    normal_events = df[df['is_flattened'] == 0].copy()
    flattened_events = df[df['is_flattened'] == 1].copy()

    # Step 4: Merge Flattened Activities
    merged_flattened = (
        flattened_events.groupby(['Case', 'Timestamp'], as_index=False)
        .agg({
            'Activity': lambda x: ';'.join(sorted(x)),
            'label': 'first' if 'label' in df.columns else lambda x: None
        })
    )

    # Step 5: Combine and Sort
    final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
    final_df = final_df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)
    final_df = final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore')

    # Step 6: Calculate Detection Metrics (if label column exists)
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
        print("No labels available for metric calculation.")
        precision = recall = f1 = 0.0

    # Step 7: Integrity Check
    total_flattened_groups = flattened_events['group_key'].nunique()
    total_flattened_events = len(flattened_events)
    total_events = len(df)
    percentage_flattened = (total_flattened_events / total_events) * 100
    print(f"Total flattened groups detected: {total_flattened_groups}")
    print(f"Total events marked as flattened: {total_flattened_events}")
    print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

    # Step 8: Save Output
    columns_to_save = ['Case', 'Activity', 'Timestamp']
    if 'label' in df.columns:
        columns_to_save.append('label')
    final_df = final_df[columns_to_save]
    final_df.to_csv(output_file, index=False)

    # Step 9: Summary Statistics
    unique_activities_before = df['Activity'].nunique()
    unique_activities_after = final_df['Activity'].nunique()
    reduction_percentage = ((total_events - len(final_df)) / total_events) * 100
    print(f"Run 3: Processed dataset saved to: {output_file}")
    print(f"Run 3: Final dataset shape: {final_df.shape}")
    print(f"Unique activities before merging: {unique_activities_before}")
    print(f"Unique activities after merging: {unique_activities_after}")
    print(f"Total reduction percentage: {reduction_percentage:.2f}%")
    print("Sample merged activities:")
    print(final_df.head(10).to_string(index=False))

except Exception as e:
    print(f"An error occurred during processing: {e}")