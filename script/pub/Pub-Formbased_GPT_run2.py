# Generated script for Pub-Formbased - Run 2
# Generated on: 2025-11-14T13:28:36.094764
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
input_file = 'data/pub/Pub-Formbased.csv'
output_file = 'data/pub/pub_form_based_cleaned_run2.csv'

try:
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")

    # Ensure required columns exist
    required_columns = ['Case', 'Activity', 'Timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Optional columns
    optional_columns = ['label', 'Variant', 'Resource']
    for col in optional_columns:
        if col not in df.columns:
            df[col] = None  # Add missing optional columns as empty

    # Convert Timestamp to datetime and standardize format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # Drop rows with invalid timestamps
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Sort by Case and Timestamp
    df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Identify flattened events
    df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp']
    group_counts = df.groupby('group_key').size()
    df['is_flattened'] = df['group_key'].map(lambda x: 1 if group_counts[x] >= 2 else 0)

    # Split dataset into normal and flattened events
    normal_events = df[df['is_flattened'] == 0].copy()
    flattened_events = df[df['is_flattened'] == 1].copy()

    # Merge flattened activities
    merged_flattened = (
        flattened_events.groupby(['Case', 'Timestamp'])
        .agg({
            'Activity': lambda x: ';'.join(sorted(x.unique())),
            'label': 'first',  # Keep the first label if present
            'Variant': 'first',
            'Resource': 'first'
        })
        .reset_index()
    )

    # Combine normal and merged flattened events
    final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
    final_df = final_df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

    # Drop helper columns
    final_df = final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore')

    # Calculate detection metrics if label column exists
    if 'label' in df.columns and df['label'].notna().any():
        y_true = df['label'].notna().astype(int)
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
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")
        print("No labels available for metric calculation.")

    # Integrity check
    total_flattened_groups = len(flattened_events['group_key'].unique())
    total_flattened_events = len(flattened_events)
    percentage_flattened = (total_flattened_events / len(df)) * 100
    print(f"Total flattened groups detected: {total_flattened_groups}")
    print(f"Total events marked as flattened: {total_flattened_events}")
    print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

    # Save the final dataset
    final_df.to_csv(output_file, index=False)

    # Summary statistics
    print(f"Run 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {final_df.shape}")
    print(f"Run 2: Dataset: pub")
    print(f"Run 2: Task type: form_based")
    print(f"Run 2: Total events: {len(df)}")
    print(f"Run 2: Flattened (merged) events detected: {total_flattened_events}")
    print(f"Run 2: Unique activities before merging: {df['Activity'].nunique()}")
    print(f"Run 2: Unique activities after merging: {final_df['Activity'].nunique()}")
    print(f"Run 2: Total reduction percentage: {100 * (1 - len(final_df) / len(df)):.2f}%")
    print("Sample of merged activities:")
    print(final_df.head(10))

except Exception as e:
    print(f"An error occurred: {e}")