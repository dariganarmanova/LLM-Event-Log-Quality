# Generated script for Pub-Polluted - Run 3
# Generated on: 2025-11-14T13:37:26.698736
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

# Additional parameters
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Input and output file paths
input_file = 'data/pub/Pub-Polluted.csv'
output_file = 'data/pub/pub_polluted_cleaned_run3.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Ensure required columns exist
required_columns = ['Activity']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required column(s): {', '.join(required_columns)}")

# Normalize column names
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Store original activities for reference
df['original_activity'] = df['Activity']

# Step 2: Define aggressive normalization function
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
print(f"Run 3: Unique base activities count: {df['BaseActivity'].nunique()}")

# Step 4: Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('original_activity', 'nunique'),
    total_count=('original_activity', 'size')
).reset_index()

# Identify polluted base activities
polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Step 5: Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 3: Polluted events count: {polluted_count}")
print(f"Run 3: Clean events count: {clean_count}")
print(f"Run 3: Pollution rate: {pollution_rate:.2f}%")

# Step 6: Calculate detection metrics
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

# Step 7: Integrity check
assert polluted_count == df[df['is_polluted_label'] == 1].shape[0], "Mismatch in polluted event counts"

# Step 8: Fix activities
df['Activity'] = df.apply(
    lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1
)

# Step 9: Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary statistics
labels_replaced = polluted_count
replacement_rate = (labels_replaced / len(df)) * 100
unique_before = df['original_activity'].nunique()
unique_after = df_fixed['Activity'].nunique()
reduction_count = unique_before - unique_after
reduction_percentage = (reduction_count / unique_before) * 100

print(f"Run 3: Normalization strategy: {normalization_strategy}")
print(f"Run 3: Total rows: {len(df)}")
print(f"Run 3: Labels replaced count: {labels_replaced}")
print(f"Run 3: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 3: Unique activities before: {unique_before} → after: {unique_after}")
print(f"Run 3: Activity reduction count: {reduction_count} ({reduction_percentage:.2f}%)")
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
sample_changes = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].head(10)
print("Run 3: Sample transformations (original → fixed):")
print(sample_changes.to_string(index=False))