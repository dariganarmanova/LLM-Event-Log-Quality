# Generated script for Credit-Polluted - Run 1
# Generated on: 2025-11-13T16:38:28.169199
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
import string

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

# Ensure the required column 'Activity' exists; raise an error if missing
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' missing")

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
    activity = re.sub('['+string.punctuation+']', ' ', activity)
    
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r'\w\d+\w', '', activity)
    
    # Remove long digit strings
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace
    activity = re.sub('\s+', ' ', activity).strip()
    
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
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().gt(min_matching_events).reset_index()[['BaseActivity']]
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases['BaseActivity'])

# Print: Number of polluted groups found
print(f"Number of polluted groups found: {polluted_bases.shape[0]}")

# Flag polluted events
polluted_count = df['is_polluted_label'].sum()
clean_count = df[~df['is_polluted_label']].shape[0]
pollution_rate = (polluted_count / df.shape[0]) * 100

# Print: Polluted events count
print(f"Polluted events count: {polluted_count}")

# Print: Clean events count
print(f"Clean events count: {clean_count}")

# Print: Pollution rate (% of total)
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna()
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
    print("No ground-truth labels found, skipping evaluation")

# Integrity check
total_polluted = df['is_polluted_label'].sum()
total_clean = df[~df['is_polluted_label']].shape[0]
assert total_polluted + total_clean == df.shape[0]
assert df[df['is_polluted_label'] == False].shape[0] == total_clean
assert df[df['is_polluted_label'] == True].shape[0] == total_polluted

# Fix activities
df.loc[df['is_polluted_label'], 'Activity'] = df.loc[df['is_polluted_label'], 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_file = f"data/credit/credit_polluted_cleaned_run1.csv"
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Normalization strategy name: aggressive")
print(f"Total rows: {df.shape[0]}")
print(f"Labels replaced count: {df[df['is_polluted_label']].shape[0]}")
print(f"Replacement rate: {(df[df['is_polluted_label']].shape[0] / df.shape[0]) * 100:.2f}%")
print(f"Unique activities before → after: {df['Activity'].nunique()} → {df_fixed['Activity'].nunique()}")
print(f"Activity reduction count and percentage: {df.shape[0] - df_fixed.shape[0]} ({(df.shape[0] - df_fixed.shape[0]) / df.shape[0] * 100:.2f}%)")
print(f"Output file path: {output_file}")

# Print up to 10 sample transformations
sample_transformations = df[df['is_polluted_label']].head(10)[['original_activity', 'Activity']]
print(sample_transformations)