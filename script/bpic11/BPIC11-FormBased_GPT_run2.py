# Generated script for BPIC11-FormBased - Run 2
# Generated on: 2025-11-13T11:17:02.445627
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
output_file = 'data/bpic11/bpic11_form_based_cleaned_run2.csv'

try:
    # Step 1: Load CSV
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = ['Case', 'Activity', 'Timestamp']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    # Convert Timestamp to datetime and standardize format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    if df['Timestamp'].isnull().any():
        raise ValueError("Invalid or missing timestamps detected.")
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
    final_df = pd.concat([normal_events[['Case', 'Activity', 'Timestamp', 'label']] if 'label' in df.columns else normal_events[['Case', 'Activity', 'Timestamp']],
                          merged_flattened], ignore_index=True)
    final_df = final_df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Step 6: Calculate Detection Metrics (if label column exists)
    if 'label' in df.columns:
        y_true = df['label'].notnull().astype(int)
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
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")
        print("No labels available for metric calculation.")

    # Step 7: Integrity Check
    total_flattened_groups = flattened_events['group_key'].nunique()
    total_flattened_events = len(flattened_events)
    total_events = len(df)
    flattened_percentage = (total_flattened_events / total_events) * 100
    print("\n=== Integrity Check ===")
    print(f"Total flattened groups detected: {total_flattened_groups}")
    print(f"Total events marked as flattened: {total_flattened_events}")
    print(f"Percentage of flattened events: {flattened_percentage:.2f}%")

    # Step 8: Save Output
    final_df.to_csv(output_file, index=False)
    print(f"\nRun 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {final_df.shape}")

    # Step 9: Summary Statistics
    unique_activities_before = df['Activity'].nunique()
    unique_activities_after = final_df['Activity'].nunique()
    total_reduction = ((len(df) - len(final_df)) / len(df)) * 100
    print("\n=== Summary Statistics ===")
    print(f"Total number of events: {len(df)}")
    print(f"Number of flattened (merged) events detected: {total_flattened_events}")
    print(f"Number of unique activities before merging: {unique_activities_before}")
    print(f"Number of unique activities after merging: {unique_activities_after}")
    print(f"Total reduction percentage: {total_reduction:.2f}%")
    print(f"Sample of merged activities:")
    print(final_df['Activity'].head(10).to_string(index=False))

except Exception as e:
    print(f"An error occurred during processing: {e}")