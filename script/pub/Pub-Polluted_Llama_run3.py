# Generated script for Pub-Polluted - Run 3
# Generated on: 2025-11-14T13:38:22.179256
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
input_file = 'data/pub/Pub-Polluted.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
output_suffix = '_fixed'

# Available columns
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
variant_column = 'Variant'
resource_column = 'Resource'

# Parameters
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Run number
run_number = 3

# Load the data
df = pd.read_csv(input_file)

# Ensure the required column 'Activity' exists; raise an error if missing
if target_column not in df.columns:
    raise ValueError(f"Missing required column '{target_column}'")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df[activity_column]

# Print original dataset shape
print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Print number of unique activities before fixing
print(f"Run {run_number}: Unique activities before fixing: {df[activity_column].nunique()}")

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

# Drop rows with empty 'BaseActivity' (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print unique base activities count
print(f"Run {run_number}: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['original_activity'].nunique().gt(min_variants).index

# Print number of polluted groups found
print(f"Run {run_number}: Polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases)

# Print polluted events count
print(f"Run {run_number}: Polluted events count: {df['is_polluted_label'].sum()}")

# Print clean events count
print(f"Run {run_number}: Clean events count: {df['is_polluted_label'].value_counts()[0]}")

# Print pollution rate (% of total)
print(f"Run {run_number}: Pollution rate: {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
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
total_events_flagged = df['is_polluted_label'].sum()
total_clean_events = len(df) - total_events_flagged
print(f"Run {run_number}: Total polluted bases detected: {total_polluted_bases}")
print(f"Run {run_number}: Total events flagged as polluted: {total_events_flagged}")
print(f"Run {run_number}: Total clean events: {total_clean_events}")

# Fix activities
df.loc[df['is_polluted_label'], target_column] = df.loc[df['is_polluted_label'], 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(f'{input_directory}/{dataset_name}{output_suffix}.csv', index=False)

# Print summary
print(f"Run {run_number}: Processed dataset saved to: {input_directory}/{dataset_name}{output_suffix}.csv")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {label_column}")

# Print up to 10 sample transformations
print("Run {run_number}: Sample transformations:")
for i in range(min(10, len(df))):
    print(f"{df.loc[i, 'original_activity']} → {df.loc[i, target_column]}")