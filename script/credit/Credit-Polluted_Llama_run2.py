# Generated script for Credit-Polluted - Run 2
# Generated on: 2025-11-13T16:38:29.728506
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Input configuration
input_file = 'data/credit/Credit-Polluted.csv'
input_directory = 'data/credit'
dataset_name = 'credit'
output_suffix = '_fixed'

# Available columns
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Optional columns
variant_column = 'Variant'
resource_column = 'Resource'

# Parameters
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names for 'Case'/ 'CaseID' if present
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df[activity_column]

# Print number of unique activities before fixing
print(f"Run 2: Unique activities before fixing: {df[activity_column].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    # Lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r'\b\w\d+\b', '', activity)
    
    # Remove long digit strings
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting
    activity = ' '.join(activity.split()[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df[activity_column].apply(aggressive_normalize)
df = df.dropna(subset=['BaseActivity'])  # Drop rows with empty BaseActivity
print(f"Run 2: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['original_activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['original_activity'] > min_variants]['BaseActivity'].tolist()
print(f"Run 2: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
print(f"Run 2: Polluted events count: {df['is_polluted_label'].sum()}")
print(f"Run 2: Clean events count: {df['is_polluted_label'].value_counts()[0]}")
print(f"Run 2: Pollution rate: {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = (y_true & y_pred).sum() / (y_pred.sum() + 1e-6)
    recall = (y_true & y_pred).sum() / (y_true.sum() + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df['is_polluted_label'].sum()
total_events_clean = df['is_polluted_label'].value_counts()[0]
assert total_events_flagged == total_polluted_bases
assert total_events_clean == len(df) - total_events_flagged

# Fix activities
df.loc[df['is_polluted_label'] == 1, activity_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(f'{input_directory}/{dataset_name}{output_suffix}.csv', index=False)
print(f"Run 2: Processed dataset saved to: {input_directory}/{dataset_name}{output_suffix}.csv")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: {target_column}")

# Print summary statistics
print(f"Run 2: Normalization strategy name: {normalization_strategy}")
print(f"Run 2: Total rows: {df_fixed.shape[0]}")
print(f"Run 2: Labels replaced count: {df_fixed[activity_column].value_counts()[0]}")
print(f"Run 2: Replacement rate: {(df_fixed[activity_column].value_counts()[0] / df.shape[0]) * 100:.2f}%")
print(f"Run 2: Unique activities before → after: {df[activity_column].nunique()} → {df_fixed[activity_column].nunique()}")
print(f"Run 2: Activity reduction count and percentage: {df[activity_column].nunique() - df_fixed[activity_column].nunique()} ({((df[activity_column].nunique() - df_fixed[activity_column].nunique()) / df[activity_column].nunique()) * 100:.2f}%)")
print(f"Run 2: Output file path: {input_directory}/{dataset_name}{output_suffix}.csv")

# Print sample transformations
if df_fixed.shape[0] > 10:
    print("Run 2: Sample transformations:")
    for i in range(10):
        print(f"{df.loc[i, activity_column]} → {df_fixed.loc[i, activity_column]}")
else:
    print("Run 2: No sample transformations available.")