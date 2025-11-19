# Generated script for BPIC11-Polluted - Run 3
# Generated on: 2025-11-13T11:44:26.658856
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
input_file = 'data/bpic11/BPIC11-Polluted.csv'
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print: Original dataset shape
print(f"Run 3: Original dataset shape: {df.shape}")

# Print: Number of unique activities before fixing
print(f"Run 3: Unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
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
    tokens = activity.split()
    if len(tokens) > aggressive_token_limit:
        activity = ' '.join(tokens[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df.dropna(subset=['BaseActivity'])  # Drop rows with empty 'BaseActivity'
print(f"Run 3: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['Activity'] > min_variants]['BaseActivity'].tolist()
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
print(f"Run 3: Polluted events count: {df['is_polluted_label'].sum()}")
print(f"Run 3: Clean events count: {df['is_polluted_label'].value_counts()[0]}")
print(f"Run 3: Pollution rate (% of total): {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = np.mean(y_true[y_pred == 1])
    recall = np.mean(y_true[y_pred == 1])
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else f"✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity check
polluted_count = df['is_polluted_label'].sum()
clean_count = df['is_polluted_label'].value_counts()[0]
total_count = len(df)
print(f"Run 3: Total polluted bases detected: {len(polluted_bases)}")
print(f"Run 3: Total events flagged as polluted: {polluted_count}")
print(f"Run 3: Total clean events: {clean_count}")
print(f"Run 3: Verification: Only events in polluted bases are set to change.")

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv('data/bpic11/bpic11_polluted_cleaned_run3.csv', index=False)

# Summary statistics
print(f"Run 3: Normalization strategy name: aggressive")
print(f"Run 3: Total rows: {len(df_fixed)}")
print(f"Run 3: Labels replaced count: {df['is_polluted_label'].sum()}")
print(f"Run 3: Replacement rate (%): {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")
print(f"Run 3: Unique activities before → after: {df['Activity'].nunique()} → {df_fixed['Activity'].nunique()}")
print(f"Run 3: Activity reduction count and percentage: {df['Activity'].nunique() - df_fixed['Activity'].nunique()} ({((df['Activity'].nunique() - df_fixed['Activity'].nunique()) / df['Activity'].nunique()) * 100:.2f}%)")
print(f"Run 3: Output file path: data/bpic11/bpic11_polluted_cleaned_run3.csv")

# Print up to 10 sample transformations
sample_transformations = df.loc[df['is_polluted_label'] == 1, ['original_activity', 'Activity']].head(10)
print(sample_transformations)