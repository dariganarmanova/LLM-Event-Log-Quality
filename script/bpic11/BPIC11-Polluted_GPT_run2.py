# Generated script for BPIC11-Polluted - Run 2
# Generated on: 2025-11-13T11:45:11.396429
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Polluted.csv'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run2.csv'
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

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")
except Exception as e:
    raise Exception(f"Error loading input file: {e}")

# Ensure required columns exist
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' is missing from the dataset.")
if 'Case' not in df.columns:
    if 'CaseID' in df.columns:
        df.rename(columns={'CaseID': 'Case'}, inplace=True)
    else:
        raise ValueError("Required column 'Case' or 'CaseID' is missing from the dataset.")

# Store original activities for reference
df['original_activity'] = df['Activity']

# Print initial stats
print(f"Run 2: Number of unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ""
    activity = activity.lower()
    activity = re.sub(r"[_\-\.,;:]", " ", activity)
    activity = re.sub(r"\b\w*\d+\w*\b", "", activity)  # Remove tokens with digits
    activity = re.sub(r"\b\d{5,}\b", "", activity)  # Remove long digit strings
    activity = re.sub(r"\s+", " ", activity).strip()  # Collapse whitespace
    tokens = activity.split()[:aggressive_token_limit]
    return " ".join(tokens)

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ""]  # Drop rows with empty BaseActivity
print(f"Run 2: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('original_activity', 'nunique'),
    total_count=('original_activity', 'size')
).reset_index()

polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 2: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 2: Polluted events count: {polluted_count}")
print(f"Run 2: Clean events count: {clean_count}")
print(f"Run 2: Pollution rate: {pollution_rate:.2f}%")

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
    print("No ground-truth labels found, skipping evaluation.")
    precision = recall = f1 = 0.0

# Integrity check
assert polluted_count == df[df['is_polluted_label'] == 1].shape[0], "Pollution flagging mismatch detected."

# Fix activities
df['Activity'] = df.apply(
    lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1
)

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Summary statistics
labels_replaced = polluted_count
replacement_rate = (labels_replaced / len(df)) * 100
unique_before = df['original_activity'].nunique()
unique_after = df_fixed['Activity'].nunique()
reduction_count = unique_before - unique_after
reduction_percentage = (reduction_count / unique_before) * 100

print(f"Run 2: Normalization strategy: {normalization_strategy}")
print(f"Run 2: Total rows: {len(df)}")
print(f"Run 2: Labels replaced count: {labels_replaced}")
print(f"Run 2: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 2: Unique activities before: {unique_before} → after: {unique_after}")
print(f"Run 2: Activity reduction count: {reduction_count} ({reduction_percentage:.2f}%)")
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
sample_transformed = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].head(10)
if not sample_transformed.empty:
    print("Run 2: Sample transformations (original → fixed):")
    print(sample_transformed.to_string(index=False))