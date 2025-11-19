# Generated script for Credit-Polluted - Run 2
# Generated on: 2025-11-13T16:30:24.456739
# Model: deepseek-ai/DeepSeek-V3-0324

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

# Task parameters
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load and validate data
df = pd.read_csv('data/credit/Credit-Polluted.csv')
print(f"Run 2: Original dataset shape: {df.shape}")
print(f"Run 2: Unique activities before fixing: {df[target_column].nunique()}")

# Store original activities
df['original_activity'] = df[target_column].copy()

# Normalize column names
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Lowercase
    s = str(activity).lower()
    # Replace punctuation with spaces
    s = re.sub(r'[_\-\.,;:]', ' ', s)
    # Remove alphanumeric ID-like tokens
    s = ' '.join([token for token in s.split() if not any(c.isdigit() for c in token)])
    # Remove long digit strings (5+ digits)
    s = re.sub(r'\b\d{5,}\b', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Token limiting
    tokens = s.split()[:aggressive_token_limit]
    return ' '.join(tokens)

# Apply normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
# Drop rows with empty BaseActivity
df = df[df['BaseActivity'] != ''].copy()
print(f"Run 2: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Detect polluted groups
group_stats = df.groupby('BaseActivity')[target_column].agg(['nunique', 'count']).reset_index()
group_stats.columns = ['BaseActivity', 'unique_variants', 'total_count']
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 2: Polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 2: Polluted events count: {polluted_count}")
print(f"Run 2: Clean events count: {clean_count}")
print(f"Run 2: Pollution rate: {pollution_rate:.2f}%")

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
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation")

# Integrity check
print(f"Run 2: Integrity check - Total polluted bases: {len(polluted_bases)}")
print(f"Run 2: Integrity check - Total events flagged as polluted: {polluted_count}")
print(f"Run 2: Integrity check - Total clean events: {clean_count}")

# Fix activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_path = 'data/credit/credit_polluted_cleaned_run2.csv'
df_fixed.to_csv(output_path, index=False)

# Summary statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df)) * 100
activity_reduction = unique_activities_before - unique_activities_after
reduction_percentage = (activity_reduction / unique_activities_before) * 100

print("\n=== Summary Statistics ===")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Activity reduction: {activity_reduction} ({reduction_percentage:.2f}%)")
print(f"Output file path: {output_path}")

# Sample transformations
sample_changes = df[df['is_polluted_label'] == 1].sample(min(10, polluted_count), random_state=2)[['original_activity', target_column]]
print("\nSample transformations:")
for _, row in sample_changes.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

# Required final prints
print(f"\nRun 2: Processed dataset saved to: {output_path}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: polluted")