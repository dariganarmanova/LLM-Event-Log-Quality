# Generated script for Credit-Polluted - Run 1
# Generated on: 2025-11-13T16:29:37.053248
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Task-specific parameters
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load the data
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run1.csv'
df = pd.read_csv(input_file)

# Validate required columns
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the input file")

# Normalize column names
column_map = {}
for col in df.columns:
    if col.lower() == 'caseid':
        column_map[col] = 'Case'
    elif col.lower() == 'case':
        column_map[col] = 'Case'
df.rename(columns=column_map, inplace=True)

# Store original activities
df['original_activity'] = df[target_column].copy()

# Print initial stats
print(f"Run 1: Original dataset shape: {df.shape}")
print(f"Run 1: Unique activities before fixing: {df[target_column].nunique()}")

# Aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Lowercase
    s = str(activity).lower()
    # Replace punctuation with spaces
    s = re.sub(r'[_\-.,;:]', ' ', s)
    # Remove alphanumeric ID-like tokens
    s = ' '.join([token for token in s.split() if not any(c.isdigit() for c in token)])
    # Remove long digit strings (5+ digits)
    s = re.sub(r'\d{5,}', '', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    # Token limiting
    tokens = s.split()[:aggressive_token_limit]
    return ' '.join(tokens).strip()

# Apply normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
# Drop rows with empty BaseActivity
df = df[df['BaseActivity'] != ''].copy()
print(f"Run 1: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Detect polluted groups
group_stats = df.groupby('BaseActivity')[target_column].agg(['nunique', 'count']).rename(columns={'nunique': 'unique_variants', 'count': 'total_count'})
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants].index.tolist()
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 1: Polluted events count: {polluted_count}")
print(f"Run 1: Clean events count: {clean_count}")
print(f"Run 1: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics
if label_column in df.columns:
    y_true = (~df[label_column].isna() & (df[label_column] != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓" if precision >= 0.6 else "✗", "Precision threshold (≥ 0.6) met" if precision >= 0.6 else "not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation")

# Integrity check
print(f"Run 1: Integrity check - Total polluted bases: {len(polluted_bases)}")
print(f"Run 1: Integrity check - Total events flagged as polluted: {polluted_count}")
print(f"Run 1: Integrity check - Total clean events: {clean_count}")

# Fix activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
output_columns = [col for col in df.columns if col not in ['original_activity', 'BaseActivity', 'is_polluted_label']]
df_fixed = df[output_columns].copy()
df_fixed.to_csv(output_file, index=False)

# Summary statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df_fixed[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df_fixed)) * 100
activity_reduction = unique_activities_before - unique_activities_after
activity_reduction_pct = (activity_reduction / unique_activities_before) * 100

print(f"\nRun 1: Summary Statistics")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Activity reduction: {activity_reduction} ({activity_reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed_rows = df[df['is_polluted_label'] == 1].head(10)
if not changed_rows.empty:
    print("\nSample transformations (original → fixed):")
    for _, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")
else:
    print("\nNo sample transformations to show (no polluted rows found)")

# Final printouts
print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: polluted")