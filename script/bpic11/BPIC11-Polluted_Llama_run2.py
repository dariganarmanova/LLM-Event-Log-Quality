# Generated script for BPIC11-Polluted - Run 2
# Generated on: 2025-11-13T11:44:23.671081
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
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

# Load the data
input_file = 'data/bpic11/BPIC11-Polluted.csv'
df = pd.read_csv(input_file)

# Ensure the required column 'Activity' exists; raise an error if missing
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' is missing from the input CSV file.")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print: Original dataset shape
print(f"Original dataset shape: {df.shape}")

# Print: Number of unique activities before fixing
print(f"Number of unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    # Lowercase: convert to lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces: `_ - . , ; :` → space (regex)
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens: tokens containing digits (e.g., `user45`, `step03`, `A1`)
    activity = re.sub(r'\b\w*\d+\w*\b', '', activity)
    
    # Remove long digit strings: remove sequences of 5+ digits
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace: trim and reduce multiple spaces to single
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting: keep only the first `aggressive_token_limit` tokens
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
polluted_bases = polluted_bases[polluted_bases['Activity'] > 2].set_index('BaseActivity')['Activity']

# Collect all polluted base keys into 'polluted_bases'
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)

# Print: Number of polluted groups found
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
polluted_events = df[df['is_polluted_label'] == 1].shape[0]
clean_events = df[df['is_polluted_label'] == 0].shape[0]
pollution_rate = (polluted_events / df.shape[0]) * 100

# Print: Polluted events count
print(f"Polluted events count: {polluted_events}")

# Print: Clean events count
print(f"Clean events count: {clean_events}")

# Print: Pollution rate (% of total)
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Print: Detection Performance Metrics
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print: Precision threshold (≥ 0.6) met/not met
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity Check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df[df['is_polluted_label'] == 1].shape[0]
total_clean_events = df[df['is_polluted_label'] == 0].shape[0]

# Confirm that only events in polluted bases are set to change
assert total_events_flagged == df[df['BaseActivity'].isin(polluted_bases)].shape[0]

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df[df['is_polluted_label'] == 1]['BaseActivity']

# Save fixed output
output_file = f'data/bpic11/bpic11_polluted_cleaned_run2.csv'
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Print summary
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: polluted")

# Print sample transformations
sample_transformations = df[df['is_polluted_label'] == 1][['original_activity', 'Activity']].head(10)
print(f"Sample transformations (first 10 rows):")
print(sample_transformations)

# Print unique activities **before** → **after**
unique_activities_before = df['Activity'].nunique()
unique_activities_after = df_fixed['Activity'].nunique()
activity_reduction_count = unique_activities_before - unique_activities_after
activity_reduction_percentage = (activity_reduction_count / unique_activities_before) * 100

# Print: Normalization strategy name
print(f"Normalization strategy name: aggressive")

# Print: Total rows
print(f"Total rows: {df.shape[0]}")

# Print: Labels replaced count
print(f"Labels replaced count: {df[df['is_polluted_label'] == 1].shape[0]}")

# Print: Replacement rate (%)
print(f"Replacement rate: {pollution_rate:.2f}%")

# Print: Output file path
print(f"Output file path: {output_file}")

# Print: Activity reduction count and percentage
print(f"Activity reduction count: {activity_reduction_count}")
print(f"Activity reduction percentage: {activity_reduction_percentage:.2f}%")