# Generated script for Pub-Polluted - Run 2
# Generated on: 2025-11-14T13:37:11.830643
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/pub/Pub-Polluted.csv'
output_file = 'data/pub/pub_polluted_cleaned_run2.csv'
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
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")
except Exception as e:
    raise Exception(f"Error loading file: {e}")

# Ensure required columns exist
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' is missing from the dataset.")

# Normalize column names for Case/CaseID
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Store original activities for reference
df['original_activity'] = df['Activity']

# Print number of unique activities before fixing
print(f"Run 2: Unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ""
    # Step 1: Lowercase
    activity = activity.lower()
    # Step 2: Replace punctuation with spaces
    activity = re.sub(r"[_\-\.,;:]", " ", activity)
    # Step 3: Remove alphanumeric ID-like tokens
    activity = re.sub(r"\b\w*\d+\w*\b", "", activity)
    # Step 4: Remove long digit sequences (5+ digits)
    activity = re.sub(r"\b\d{5,}\b", "", activity)
    # Step 5: Collapse whitespace
    activity = re.sub(r"\s+", " ", activity).strip()
    # Step 6: Token limiting
    tokens = activity.split()
    activity = " ".join(tokens[:aggressive_token_limit])
    return activity

# Apply aggressive normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty BaseActivity
df = df[df['BaseActivity'] != ""]
print(f"Run 2: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity')['Activity'].agg(['nunique', 'count']).reset_index()
grouped.rename(columns={'nunique': 'unique_variants', 'count': 'total_count'}, inplace=True)
polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 2: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 2: Polluted events count: {polluted_count}")
print(f"Run 2: Clean events count: {clean_count}")
print(f"Run 2: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics if label column exists
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
assert polluted_count == len(df[df['is_polluted_label'] == 1]), "Mismatch in polluted event counts."

# Fix activities
df['Activity'] = df.apply(lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1)

# Drop helper columns
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])

# Save the processed data
df_fixed.to_csv(output_file, index=False)

# Print summary statistics
print(f"Run 2: Normalization strategy: {normalization_strategy}")
print(f"Run 2: Total rows: {len(df_fixed)}")
print(f"Run 2: Labels replaced count: {polluted_count}")
print(f"Run 2: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 2: Unique activities before: {df['original_activity'].nunique()} → after: {df_fixed['Activity'].nunique()}")
print(f"Run 2: Activity reduction: {df['original_activity'].nunique() - df_fixed['Activity'].nunique()} ({((df['original_activity'].nunique() - df_fixed['Activity'].nunique()) / df['original_activity'].nunique()) * 100:.2f}%)")
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
sample_changes = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].head(10)
print("Run 2: Sample transformations:")
print(sample_changes.to_string(index=False))