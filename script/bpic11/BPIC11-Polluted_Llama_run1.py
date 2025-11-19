# Generated script for BPIC11-Polluted - Run 1
# Generated on: 2025-11-13T11:44:21.371256
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

# Load the data
input_file = 'data/bpic11/BPIC11-Polluted.csv'
df = pd.read_csv(input_file)

# Ensure the required column 'Activity' exists; raise an error if missing
if 'Activity' not in df.columns:
    raise ValueError("The required column 'Activity' is missing.")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print: Original dataset shape
print(f"Original dataset shape: {df.shape}")

# Print: Number of unique activities before fixing
print(f"Unique activities before fixing: {df['Activity'].nunique()}")

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
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting
    tokens = activity.split()
    if len(tokens) > 3:
        activity = ' '.join(tokens[:3])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty 'BaseActivity' (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print: Unique base activities count
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['Activity'] > 2]['BaseActivity'].tolist()

# Print: Number of polluted groups found
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)

# Print:
#  * Polluted events count
#  * Clean events count
#  * Pollution rate (% of total)
polluted_count = (df['is_polluted_label'] == 1).sum()
clean_count = (df['is_polluted_label'] == 0).sum()
pollution_rate = (polluted_count / len(df)) * 100
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = (y_true & y_pred).sum() / (y_pred.sum() + 1e-9)
    recall = (y_true & y_pred).sum() / (y_true.sum() + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else f"✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity check
total_polluted = df['BaseActivity'].isin(polluted_bases).sum()
total_polluted_events = (df['is_polluted_label'] == 1).sum()
total_clean_events = (df['is_polluted_label'] == 0).sum()
assert total_polluted == total_polluted_events
assert total_polluted + total_clean_events == len(df)

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_file = f"data/bpic11/bpic11_polluted_cleaned_run1.csv"
df_fixed.to_csv(output_file, index=False)

# Print summary
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: polluted")

# Print unique activities before → after
unique_before = df['Activity'].nunique()
unique_after = df_fixed['Activity'].nunique()
activity_reduction_count = unique_before - unique_after
activity_reduction_percentage = (activity_reduction_count / unique_before) * 100
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction count: {activity_reduction_count}")
print(f"Activity reduction percentage: {activity_reduction_percentage:.2f}%")

# Print up to 10 sample transformations
sample_transformations = df.loc[df['is_polluted_label'] == 1, ['original_activity', 'Activity']].head(10)
print(sample_transformations)