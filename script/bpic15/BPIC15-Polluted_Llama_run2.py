# Generated script for BPIC15-Polluted - Run 2
# Generated on: 2025-11-13T12:46:42.150789
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Polluted.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
output_suffix = '_polluted_cleaned_run2'
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
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure the required column 'Activity' exists; raise an error if missing
if target_column not in df.columns:
    raise ValueError(f"Missing required column '{target_column}'")

# Normalize column names for 'Case'/ 'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df[target_column]

# Print: Original dataset shape
print(f"Run 2: Original dataset shape: {df.shape}")

# Print: Number of unique activities before fixing
print(f"Run 2: Unique activities before fixing: {df[target_column].nunique()}")

# Define Aggressive Normalization
def aggressive_normalize(activity):
    # Lowercase: convert to lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces: `_ - . , ; :` → space (regex)
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens: tokens containing digits (e.g., `user45`, `step03`, `A1`)
    activity = re.sub(r'\b\w*[0-9]\w*\b', '', activity)
    
    # Remove long digit strings: remove sequences of 5+ digits
    activity = re.sub(r'\b\d{5,}\b', '', activity)
    
    # Collapse whitespace: trim and reduce multiple spaces to single
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting: keep only the first `aggressive_token_limit` tokens
    activity = ' '.join(activity.split()[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply Normalization 
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)

# Drop rows with empty `BaseActivity` (nothing meaningful left to compare)
df = df.dropna(subset=['BaseActivity'])

# Print: Unique base activities count
print(f"Run 2: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
polluted_bases = df.groupby('BaseActivity')['original_activity'].nunique().reset_index()
polluted_bases = polluted_bases[polluted_bases['original_activity'] > min_variants]['BaseActivity'].tolist()

# Print: Number of polluted groups found
print(f"Run 2: Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases)

# Print: Polluted events count
print(f"Run 2: Polluted events count: {df['is_polluted_label'].sum()}")

# Print: Clean events count
print(f"Run 2: Clean events count: {df['is_polluted_label'].value_counts()[0]}")

# Print: Pollution rate (% of total)
print(f"Run 2: Pollution rate: {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
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
    print(f"No ground-truth labels found, skipping evaluation")

# Integrity Check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df['is_polluted_label'].sum()
total_events_clean = df['is_polluted_label'].value_counts()[0]
print(f"Run 2: Total polluted bases detected: {total_polluted_bases}")
print(f"Run 2: Total events flagged as polluted: {total_events_flagged}")
print(f"Run 2: Total clean events: {total_events_clean}")
print(f"Run 2: Only events in polluted bases are set to change")

# Fix Activities
df.loc[df['is_polluted_label'], target_column] = df.loc[df['is_polluted_label'], 'BaseActivity']

# Save Fixed Output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(f'{input_directory}/{dataset_name}{output_suffix}.csv', index=False)

# Summary Statistics
print(f"Run 2: Normalization strategy name: {normalization_strategy}")
print(f"Run 2: Total rows: {len(df_fixed)}")
print(f"Run 2: Labels replaced count: {df_fixed[target_column].ne(df[target_column]).sum()}")
print(f"Run 2: Replacement rate: {(df_fixed[target_column].ne(df[target_column]).sum() / len(df_fixed)) * 100:.2f}%")
print(f"Run 2: Unique activities **before** → **after**: {df[target_column].nunique()} → {df_fixed[target_column].nunique()}")
print(f"Run 2: Activity reduction count and percentage: {df[target_column].ne(df_fixed[target_column]).sum()} ({(df[target_column].ne(df_fixed[target_column]).sum() / len(df)) * 100:.2f}%)")
print(f"Run 2: Output file path: {f'{input_directory}/{dataset_name}{output_suffix}.csv'}")

# Print up to 10 sample transformations** showing: `original_activity → Activity` (only for changed rows)
sample_transformations = df.loc[df[target_column].ne(df_fixed[target_column]), [target_column, 'original_activity']]
print(sample_transformations.head(10))