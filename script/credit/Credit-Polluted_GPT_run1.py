# Generated script for Credit-Polluted - Run 1
# Generated on: 2025-11-13T16:36:58.908217
# Model: gpt-4o-2024-11-20

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

# Input and output file paths
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run1.csv'

# Normalization parameters
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Validate required columns
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' is missing from the dataset.")

# Normalize column names for Case/CaseID
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Store original activities for reference
df['original_activity'] = df['Activity']

# Print initial dataset stats
print(f"Run 1: Number of unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ""
    # Step 1: Lowercase
    activity = activity.lower()
    # Step 2: Replace punctuation with spaces
    activity = re.sub(r"[_\-.,;:]", " ", activity)
    # Step 3: Remove alphanumeric ID-like tokens
    activity = re.sub(r"\b\w*\d+\w*\b", "", activity)
    # Step 4: Remove long digit sequences
    activity = re.sub(r"\b\d{5,}\b", "", activity)
    # Step 5: Collapse whitespace
    activity = re.sub(r"\s+", " ", activity).strip()
    # Step 6: Token limiting
    tokens = activity.split()
    activity = " ".join(tokens[:aggressive_token_limit])
    return activity

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ""]  # Drop rows with empty BaseActivity
print(f"Run 1: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('original_activity', 'nunique'),
    total_count=('original_activity', 'count')
).reset_index()

polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = polluted_count / len(df) * 100

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
    print("\n=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("\n=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation.")

# Fix activities
df['Activity'] = df.apply(
    lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1
)

# Drop helper columns for final output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])

# Save the cleaned dataset
try:
    df_fixed.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Print summary statistics
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: polluted")
print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Labels replaced count: {polluted_count}")
print(f"Run 1: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 1: Unique activities before → after: {df['original_activity'].nunique()} → {df_fixed['Activity'].nunique()}")
print(f"Run 1: Activity reduction count: {df['original_activity'].nunique() - df_fixed['Activity'].nunique()}")
print(f"Run 1: Activity reduction percentage: {((df['original_activity'].nunique() - df_fixed['Activity'].nunique()) / df['original_activity'].nunique()) * 100:.2f}%")

# Print sample transformations
sample_changes = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].head(10)
print("\nSample transformations (original → fixed):")
print(sample_changes.to_string(index=False))