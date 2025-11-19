# Generated script for BPIC15-Polluted - Run 1
# Generated on: 2025-11-13T12:46:40.474817
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Polluted.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
output_suffix = '_polluted_cleaned_run1'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names for 'Case'/'CaseID' if present
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df[target_column]

# Print: Number of unique activities before fixing
print(f"Run 1: Unique activities before fixing: {df[target_column].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    # Lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r'\w\d+\w', '', activity)
    
    # Remove long digit strings
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity)
    
    # Token limiting
    activity = ' '.join(activity.split()[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
df = df.dropna(subset=['BaseActivity'])  # Drop rows with empty 'BaseActivity'
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['original_activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['original_activity'] > min_variants]['BaseActivity'].tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases)
print(f"Run 1: Polluted events count: {df['is_polluted_label'].sum()}")
print(f"Run 1: Clean events count: {~df['is_polluted_label'].sum()}")
print(f"Run 1: Pollution rate: {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = df[label_column].notna()
    y_pred = df['is_polluted_label']
    precision = (y_true & y_pred).sum() / (y_pred.sum() + 1e-9)
    recall = (y_true & y_pred).sum() / (y_true.sum() + 1e-9)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else f"✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity check
total_polluted_bases = len(polluted_bases)
total_polluted_events = df['is_polluted_label'].sum()
total_clean_events = ~df['is_polluted_label'].sum()
assert total_polluted_events == df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'].sum()
assert total_clean_events == df.loc[~df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'].sum()
print(f"Run 1: Total polluted bases detected: {total_polluted_bases}")
print(f"Run 1: Total events flagged as polluted: {total_polluted_events}")
print(f"Run 1: Total clean events: {total_clean_events}")

# Fix activities
df.loc[df['is_polluted_label'], target_column] = df.loc[df['is_polluted_label'], 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(f'{input_directory}/{dataset_name}{output_suffix}.csv', index=False)
print(f"Run 1: Processed dataset saved to: {input_directory}/{dataset_name}{output_suffix}.csv")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: {dataset_name}")
print(f"Run 1: Task type: {target_column}_polluted")