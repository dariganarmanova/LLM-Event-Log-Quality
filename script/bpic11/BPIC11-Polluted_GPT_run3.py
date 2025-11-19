# Generated script for BPIC11-Polluted - Run 3
# Generated on: 2025-11-13T11:45:26.821238
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Polluted.csv'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run3.csv'
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

# Columns
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found: {input_file}")
except Exception as e:
    raise Exception(f"Error loading file: {e}")

# Ensure required column exists
if activity_column not in df.columns:
    raise ValueError(f"Required column '{activity_column}' is missing in the dataset.")

# Normalize column names
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Store original activities
df['original_activity'] = df[activity_column]

# Print unique activities before fixing
print(f"Run 3: Number of unique activities before fixing: {df[activity_column].nunique()}")

# Aggressive normalization function
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
df['BaseActivity'] = df[activity_column].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ""]  # Drop rows with empty BaseActivity
print(f"Run 3: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=(activity_column, 'nunique'),
    total_count=(activity_column, 'count')
).reset_index()

polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].apply(lambda x: 1 if x in polluted_bases else 0)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100

print(f"Run 3: Polluted events count: {polluted_count}")
print(f"Run 3: Clean events count: {clean_count}")
print(f"Run 3: Pollution rate: {pollution_rate:.2f}%")

# Detection metrics (if label column exists)
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
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

# Integrity check
assert polluted_count == df[df['is_polluted_label'] == 1].shape[0], "Mismatch in polluted event counts."

# Fix activities
df.loc[df['is_polluted_label'] == 1, activity_column] = df['BaseActivity']

# Save cleaned dataset
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Run 3: Normalization strategy: {normalization_strategy}")
print(f"Run 3: Total rows: {len(df)}")
print(f"Run 3: Labels replaced count: {polluted_count}")
print(f"Run 3: Replacement rate: {pollution_rate:.2f}%")
print(f"Run 3: Unique activities before: {df['original_activity'].nunique()} → after: {df[activity_column].nunique()}")
print(f"Run 3: Activity reduction count: {df['original_activity'].nunique() - df[activity_column].nunique()}")
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")

# Print sample transformations
sample_transforms = df[df['original_activity'] != df[activity_column]].head(10)
print("Sample transformations (original → fixed):")
for _, row in sample_transforms.iterrows():
    print(f"{row['original_activity']} → {row[activity_column]}")