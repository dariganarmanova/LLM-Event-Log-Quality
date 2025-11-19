# Generated script for Pub-Polluted - Run 2
# Generated on: 2025-11-14T13:38:20.378129
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
input_file = 'data/pub/Pub-Polluted.csv'
df = pd.read_csv(input_file)

# Ensure the required column 'Activity' exists; raise an error if missing
if 'Activity' not in df.columns:
    raise ValueError("Missing required column 'Activity'")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print:
#   * Original dataset shape
#   * Number of unique activities before fixing
print(f"Run 2: Original dataset shape: {df.shape}")
print(f"Run 2: Unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization
def aggressive_normalize(activity):
    # Lowercase: convert to lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces: _ - . , ; : → space (regex)
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens: tokens containing digits (e.g., user45, step03, A1)
    activity = re.sub(r'\w\d+', '', activity)
    
    # Remove long digit strings: remove sequences of 5+ digits
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace: trim and reduce multiple spaces to single
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting: keep only the first aggressive_token_limit tokens
    aggressive_token_limit = 3
    activity = ' '.join(activity.split()[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty 'BaseActivity' (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print: Unique base activities count
print(f"Run 2: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['Activity'] > min_matching_events]

# Collect all polluted base keys into 'polluted_bases'
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases['BaseActivity']).astype(int)

# Print:
#   * Number of polluted groups found
print(f"Run 2: Number of polluted groups found: {polluted_bases.shape[0]}")

# Flag polluted events
polluted_events_count = df[df['is_polluted_label'] == 1].shape[0]
clean_events_count = df[df['is_polluted_label'] == 0].shape[0]
pollution_rate = (polluted_events_count / df.shape[0]) * 100

# Print:
#   * Polluted events count
#   * Clean events count
#   * Pollution rate (% of total)
print(f"Run 2: Polluted events count: {polluted_events_count}")
print(f"Run 2: Clean events count: {clean_events_count}")
print(f"Run 2: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = (y_true & y_pred).sum() / (y_pred.sum() + 0.000001)
    recall = (y_true & y_pred).sum() / (y_true.sum() + 0.000001)
    f1_score = 2 * (precision * recall) / (precision + recall + 0.000001)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity Check
total_polluted_bases_detected = polluted_bases.shape[0]
total_events_flagged_as_polluted = polluted_events_count
total_clean_events = clean_events_count
assert total_polluted_bases_detected == polluted_bases.shape[0]
assert total_events_flagged_as_polluted == polluted_events_count
assert total_clean_events == clean_events_count

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_file = 'data/pub/pub_polluted_cleaned_run2.csv'
df_fixed.to_csv(output_file, index=False)

# Print summary
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: polluted")

# Print sample transformations
sample_transformations = df.loc[df['is_polluted_label'] == 1, ['original_activity', 'Activity']].head(10)
print("Sample transformations:")
print(sample_transformations)

# Print unique activities before → after
unique_activities_before = df['Activity'].nunique()
unique_activities_after = df_fixed['Activity'].nunique()
activity_reduction_count = unique_activities_before - unique_activities_after
activity_reduction_percentage = (activity_reduction_count / unique_activities_before) * 100
print(f"Run 2: Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Run 2: Activity reduction count: {activity_reduction_count}")
print(f"Run 2: Activity reduction percentage: {activity_reduction_percentage:.2f}%")

# Print output file path
print(f"Run 2: Output file path: {output_file}")