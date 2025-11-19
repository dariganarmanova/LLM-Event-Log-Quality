# Generated script for BPIC15-Polluted - Run 1
# Generated on: 2025-11-13T12:43:50.836401
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Polluted.csv'
dataset_name = 'bpic15'
output_suffix = '_cleaned_run1'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load and validate
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the dataset.")
df['original_activity'] = df[target_column]
print(f"Run 1: Number of unique activities before fixing: {df[target_column].nunique()}")

# Aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Step 1: Lowercase
    s = str(activity).lower()
    # Step 2: Replace punctuation with spaces
    s = re.sub(r'[_\-\.,;:]', ' ', s)
    # Step 3: Remove alphanumeric ID-like tokens
    tokens = [token for token in s.split() if not any(c.isdigit() for c in token)]
    # Step 4: Remove long digit strings (5+ digits)
    tokens = [token for token in tokens if not re.search(r'\d{5,}', token)]
    # Step 5: Collapse whitespace
    s = ' '.join(tokens).strip()
    # Step 6: Token limiting
    tokens = s.split()[:aggressive_token_limit]
    # Step 7: Return joined tokens
    return ' '.join(tokens)

# Apply normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ''].copy()
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=pd.NamedAgg(column='original_activity', aggfunc='nunique'),
    total_count=pd.NamedAgg(column='original_activity', aggfunc='count')
).reset_index()
polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
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
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation")

# Integrity check
print(f"Run 1: Total polluted bases detected: {len(polluted_bases)}")
print(f"Run 1: Total events flagged as polluted: {polluted_count}")
print(f"Run 1: Total clean events: {clean_count}")

# Fix activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
output_file = f"data/bpic15/bpic15_polluted_cleaned_run1.csv"
df_fixed.to_csv(output_file, index=False)

# Summary statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df)) * 100
activity_reduction = unique_activities_before - unique_activities_after
activity_reduction_pct = (activity_reduction / unique_activities_before) * 100

print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Total rows: {len(df)}")
print(f"Run 1: Labels replaced count: {replaced_count}")
print(f"Run 1: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 1: Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Run 1: Activity reduction count: {activity_reduction}")
print(f"Run 1: Activity reduction percentage: {activity_reduction_pct:.2f}%")
print(f"Run 1: Output file path: {output_file}")

# Sample transformations
changed_rows = df[df['is_polluted_label'] == 1].head(10)
for _, row in changed_rows.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: {dataset_name}")
print(f"Run 1: Task type: polluted")