# Generated script for Credit-Formbased - Run 2
# Generated on: 2025-11-13T16:18:41.443241
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
input_file = 'data/credit/Credit-Formbased.csv'
output_file = 'data/credit/credit_form_based_cleaned_run2.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
optional_columns = ['label', 'Variant', 'Resource']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert Timestamp to datetime and standardize format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])  # Drop rows with invalid timestamps
df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

# Create group_key and identify flattened events
df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp']
group_counts = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_counts) >= 2

# Split dataset into normal and flattened events
normal_events = df[df['is_flattened'] == False].copy()
flattened_events = df[df['is_flattened'] == True].copy()

# Merge flattened events
merged_events = (
    flattened_events.groupby(['Case', 'Timestamp'])
    .agg({
        'Activity': lambda x: ';'.join(sorted(x)),
        'label': 'first',  # Keep the first label if present
        **{col: 'first' for col in optional_columns if col in df.columns}
    })
    .reset_index()
)

# Combine normal and merged events
final_df = pd.concat([normal_events, merged_events], ignore_index=True)
final_df = final_df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

# Drop helper columns
final_df = final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore')

# Calculate detection metrics if label column exists
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_flattened'].astype(int)
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

# Integrity check
total_events = len(df)
flattened_groups = len(flattened_events['group_key'].unique())
flattened_events_count = len(flattened_events)
flattened_percentage = (flattened_events_count / total_events) * 100
print(f"Total events: {total_events}")
print(f"Flattened groups detected: {flattened_groups}")
print(f"Flattened events: {flattened_events_count} ({flattened_percentage:.2f}%)")

# Save the processed data
try:
    final_df.to_csv(output_file, index=False)
    print(f"Run 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {final_df.shape}")
    print(f"Run 2: Dataset: credit")
    print(f"Run 2: Task type: form_based")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Summary statistics
unique_activities_before = len(df['Activity'].unique())
unique_activities_after = len(final_df['Activity'].unique())
reduction_percentage = ((total_events - len(final_df)) / total_events) * 100
print(f"Unique activities before merging: {unique_activities_before}")
print(f"Unique activities after merging: {unique_activities_after}")
print(f"Total reduction in events: {reduction_percentage:.2f}%")
print("Sample of merged activities:")
print(final_df.head(10))