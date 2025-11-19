# Generated script for BPIC15-Polluted - Run 3
# Generated on: 2025-11-13T12:46:43.762443
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
input_file = 'data/bpic15/BPIC15-Polluted.csv'
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

# Define Aggressive Normalization
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

# Apply Normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty 'BaseActivity' (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print: Unique base activities count
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['Activity'] > min_variants]
polluted_bases = polluted_bases['BaseActivity'].tolist()

# Print: Number of polluted groups found
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)

# Print:
# Polluted events count
print(f"Polluted events count: {df['is_polluted_label'].sum()}")
# Clean events count
print(f"Clean events count: {df['is_polluted_label'].value_counts()[0]}")
# Pollution rate (% of total)
print(f"Pollution rate (% of total): {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}")

# Calculate Detection Metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Print:
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else f"✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity Check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df['is_polluted_label'].sum()
total_clean_events = df['is_polluted_label'].value_counts()[0]
assert total_polluted_bases == total_events_flagged
assert total_events_flagged + total_clean_events == len(df)

# Fix Activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save Fixed Output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv('data/bpic15/bpic15_polluted_cleaned_run3.csv', index=False)

# Summary Statistics
print(f"Normalization strategy name: aggressive")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {df_fixed['Activity'].value_counts()[0]}")
print(f"Replacement rate (%): {(df_fixed['Activity'].value_counts()[0] / len(df_fixed)) * 100:.2f}")
print(f"Unique activities before → after: {df['Activity'].nunique()} → {df_fixed['Activity'].nunique()}")
print(f"Activity reduction count and percentage: {df['Activity'].nunique() - df_fixed['Activity'].nunique()} ({(df['Activity'].nunique() - df_fixed['Activity'].nunique()) / df['Activity'].nunique() * 100:.2f}%)")
print(f"Output file path: data/bpic15/bpic15_polluted_cleaned_run3.csv")

# Print up to 10 sample transformations showing: original_activity → Activity (only for changed rows)
sample_transformations = df.loc[df['is_polluted_label'] == 1, ['original_activity', 'Activity']].head(10)
print(sample_transformations)