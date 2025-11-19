# Generated script for Pub-Polluted - Run 1
# Generated on: 2025-11-14T13:36:55.251258
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
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Input and output paths
input_file = 'data/pub/Pub-Polluted.csv'
output_file = 'data/pub/pub_polluted_cleaned_run1.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except FileNotFoundError as e:
    print(f"Error: File not found at {input_file}")
    raise e
except Exception as e:
    print(f"Error: Failed to load the dataset. {str(e)}")
    raise e

# Ensure required columns exist
if 'Activity' not in df.columns:
    raise ValueError("Error: Required column 'Activity' is missing from the dataset.")

# Normalize column names
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Preserve original activities
df['original_activity'] = df['Activity']

# Print unique activities before fixing
print(f"Run 1: Number of unique activities before fixing: {df['Activity'].nunique()}")

# Aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Lowercase
    activity = activity.lower()
    # Replace punctuation with spaces
    activity = re.sub(r'[_\-\.,;:]', ' ', activity)
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r'\b\w*\d+\w*\b', '', activity)
    # Remove long digit strings
    activity = re.sub(r'\b\d{5,}\b', '', activity)
    # Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity).strip()
    # Token limiting
    tokens = activity.split()[:aggressive_token_limit]
    return ' '.join(tokens)

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)

# Drop rows with empty BaseActivity
df = df[df['BaseActivity'] != '']
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('original_activity', 'nunique'),
    total_count=('original_activity', 'size')
).reset_index()

polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)

# Print pollution stats
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = polluted_count / len(df) * 100
print(f"Run 1: Polluted events count: {polluted_count}")
print(f"Run 1: Clean events count: {clean_count}")
print(f"Run 1: Pollution rate: {pollution_rate:.2f}%")

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
    print("No ground-truth labels found, skipping evaluation.")
    precision = recall = f1 = 0.0

# Replace polluted activities with their base form
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df['BaseActivity']

# Drop helper columns for final output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])

# Save the cleaned dataset
try:
    df_fixed.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error: Failed to save the processed dataset. {str(e)}")
    raise e

# Print summary statistics
print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Total rows: {len(df)}")
print(f"Run 1: Labels replaced count: {polluted_count}")
print(f"Run 1: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 1: Unique activities before: {df['original_activity'].nunique()} → after: {df_fixed['Activity'].nunique()}")
activity_reduction = df['original_activity'].nunique() - df_fixed['Activity'].nunique()
activity_reduction_pct = activity_reduction / df['original_activity'].nunique() * 100
print(f"Run 1: Activity reduction count: {activity_reduction} ({activity_reduction_pct:.2f}%)")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
sample_changes = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].head(10)
print("Run 1: Sample transformations (original → fixed):")
print(sample_changes.to_string(index=False))