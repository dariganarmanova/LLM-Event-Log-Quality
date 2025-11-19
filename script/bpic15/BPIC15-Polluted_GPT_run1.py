# Generated script for BPIC15-Polluted - Run 1
# Generated on: 2025-11-13T12:45:50.179836
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Polluted.csv'
output_file = 'data/bpic15/bpic15_polluted_cleaned_run1.csv'
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
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found at {input_file}. Please check the path.")

# Ensure required columns exist
required_columns = ['Case', 'Activity']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize column names
df.rename(columns={col: 'Case' for col in df.columns if col.lower() in ['case', 'caseid']}, inplace=True)

# Store original activities for reference
df['original_activity'] = df['Activity']

# Print initial unique activities
print(f"Run 1: Number of unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ""
    activity = activity.lower()  # Lowercase
    activity = re.sub(r"[_\-\.,;:]", " ", activity)  # Replace punctuation with spaces
    activity = re.sub(r"\b\w*\d+\w*\b", "", activity)  # Remove alphanumeric tokens with digits
    activity = re.sub(r"\d{5,}", "", activity)  # Remove long digit sequences
    activity = re.sub(r"\s+", " ", activity).strip()  # Collapse whitespace
    tokens = activity.split()[:aggressive_token_limit]  # Limit tokens
    return " ".join(tokens)

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ""]  # Drop rows with empty BaseActivity
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(unique_variants=('Activity', 'nunique'), total_count=('Activity', 'size'))
polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = polluted_count / len(df) * 100
print(f"Run 1: Polluted events count: {polluted_count}")
print(f"Run 1: Clean events count: {clean_count}")
print(f"Run 1: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (if label column exists)
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
assert polluted_count == df[df['is_polluted_label'] == 1].shape[0], "Mismatch in polluted event counts."

# Fix activities
df['Activity'] = df.apply(lambda row: row['BaseActivity'] if row['is_polluted_label'] == 1 else row['Activity'], axis=1)

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Print summary
print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Total rows: {len(df)}")
print(f"Run 1: Labels replaced count: {polluted_count}")
print(f"Run 1: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 1: Unique activities before: {df['original_activity'].nunique()} → after: {df['Activity'].nunique()}")
print(f"Run 1: Activity reduction count: {df['original_activity'].nunique() - df['Activity'].nunique()}")
print(f"Run 1: Activity reduction percentage: {((df['original_activity'].nunique() - df['Activity'].nunique()) / df['original_activity'].nunique()) * 100:.2f}%")
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
changed_rows = df[df['original_activity'] != df['Activity']]
print("Run 1: Sample transformations (up to 10):")
print(changed_rows[['original_activity', 'Activity']].head(10))