# Generated script for Credit-Polluted - Run 3
# Generated on: 2025-11-13T16:38:31.232786
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
import numpy as np

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
input_file = 'data/credit/Credit-Polluted.csv'
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print: Original dataset shape
# Print: Number of unique activities before fixing
print(f"Run 3: Unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    # Lowercase: convert to lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces: _ - . , ; : → space (regex)
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens: tokens containing digits (e.g., user45, step03, A1)
    activity = re.sub(r'\w\d+\w', '', activity)
    
    # Remove long digit strings: remove sequences of 5+ digits
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace: trim and reduce multiple spaces to single
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting: keep only the first aggressive_token_limit tokens
    tokens = activity.split()
    if len(tokens) > aggressive_token_limit:
        activity = ' '.join(tokens[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty 'BaseActivity' (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print: Unique base activities count
print(f"Run 3: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['Activity'] > min_matching_events]
polluted_bases = polluted_bases['BaseActivity'].tolist()

# Print: Number of polluted groups found
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)

# Print:
#   - Polluted events count
#   - Clean events count
#   - Pollution rate (% of total)
polluted_events = df[df['is_polluted_label'] == 1].shape[0]
clean_events = df[df['is_polluted_label'] == 0].shape[0]
pollution_rate = (polluted_events / df.shape[0]) * 100
print(f"Run 3: Polluted events count: {polluted_events}")
print(f"Run 3: Clean events count: {clean_events}")
print(f"Run 3: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = np.round(np.mean(y_true[y_pred == 1]), 4)
    recall = np.round(np.mean(y_true[y_pred == 1]) / np.mean(y_true), 4)
    f1_score = np.round(2 * (precision * recall) / (precision + recall), 4)
    print(f"Run 3: === Detection Performance Metrics ===")
    print(f"Run 3: Precision: {precision}")
    print(f"Run 3: Recall: {recall}")
    print(f"Run 3: F1-Score: {f1_score}")
    if precision >= 0.6:
        print(f"Run 3: Precision threshold (≥ 0.6) met")
    else:
        print(f"Run 3: Precision threshold (≥ 0.6) NOT met")
else:
    print(f"Run 3: No ground-truth labels found, skipping evaluation")

# Integrity check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df[df['is_polluted_label'] == 1].shape[0]
total_events_clean = df[df['is_polluted_label'] == 0].shape[0]
assert total_events_flagged == total_polluted_bases
assert total_events_clean + total_events_flagged == df.shape[0]

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df[df['is_polluted_label'] == 1]['BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_file = f"data/credit/credit_polluted_cleaned_run3.csv"
df_fixed.to_csv(output_file, index=False)

# Print summary
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: polluted")

# Print unique activities before → after
unique_activities_before = df['Activity'].nunique()
unique_activities_after = df_fixed['Activity'].nunique()
activity_reduction_count = unique_activities_before - unique_activities_after
activity_reduction_percentage = (activity_reduction_count / unique_activities_before) * 100
print(f"Run 3: Unique activities before: {unique_activities_before}")
print(f"Run 3: Unique activities after: {unique_activities_after}")
print(f"Run 3: Activity reduction count: {activity_reduction_count}")
print(f"Run 3: Activity reduction percentage: {activity_reduction_percentage:.2f}%")

# Print up to 10 sample transformations
sample_transformations = df.loc[df['is_polluted_label'] == 1, ['original_activity', 'Activity']].head(10)
print(f"Run 3: Sample transformations:")
print(sample_transformations)

# Print output file path
print(f"Run 3: Output file path: {output_file}")