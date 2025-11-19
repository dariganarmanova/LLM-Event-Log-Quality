# Generated script for BPIC11-Polluted - Run 1
# Generated on: 2025-11-13T11:44:57.482541
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Polluted.csv'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run1.csv'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    raise RuntimeError(f"Error loading file {input_file}: {e}")

# Validate required columns
required_columns = ['Case', 'Activity']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize column names
df.rename(columns={'CaseID': 'Case'}, inplace=True)
df['original_activity'] = df['Activity']  # Preserve original activities

# Aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ""
    # Lowercase
    activity = activity.lower()
    # Replace punctuation with spaces
    activity = re.sub(r"[_\-\.,;:]", " ", activity)
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r"\b\w*\d+\w*\b", "", activity)
    # Remove long digit strings
    activity = re.sub(r"\b\d{5,}\b", "", activity)
    # Collapse whitespace
    activity = re.sub(r"\s+", " ", activity).strip()
    # Token limiting
    tokens = activity.split()[:aggressive_token_limit]
    return " ".join(tokens)

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ""]  # Drop rows with empty BaseActivity
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity')['original_activity'].nunique().reset_index()
grouped.rename(columns={'original_activity': 'unique_variants'}, inplace=True)
polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 1: Polluted events count: {polluted_count}")
print(f"Run 1: Clean events count: {clean_count}")
print(f"Run 1: Pollution rate: {pollution_rate:.2f}%")

# Detection metrics (if label column exists)
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation")
    precision = recall = f1 = 0.0

# Integrity check
assert polluted_count == df[df['is_polluted_label'] == 1].shape[0]
assert clean_count == df[df['is_polluted_label'] == 0].shape[0]

# Fix activities
df['Activity'] = df.apply(
    lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1
)

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
try:
    df_fixed.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    raise RuntimeError(f"Error saving file {output_file}: {e}")

# Summary statistics
unique_before = df['original_activity'].nunique()
unique_after = df_fixed['Activity'].nunique()
reduction_count = unique_before - unique_after
reduction_percentage = (reduction_count / unique_before) * 100
print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Total rows: {len(df)}")
print(f"Run 1: Labels replaced count: {polluted_count}")
print(f"Run 1: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 1: Unique activities before: {unique_before} → after: {unique_after}")
print(f"Run 1: Activity reduction count: {reduction_count} ({reduction_percentage:.2f}%)")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")

# Sample transformations
sample_transforms = df[df['is_polluted_label'] == 1][['original_activity', 'Activity']].drop_duplicates().head(10)
print("Run 1: Sample transformations (original → fixed):")
print(sample_transforms.to_string(index=False))