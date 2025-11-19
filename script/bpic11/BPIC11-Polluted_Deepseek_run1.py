# Generated script for BPIC11-Polluted - Run 1
# Generated on: 2025-11-13T11:46:40.786909
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Polluted.csv'
dataset_name = 'bpic11'
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
    raise ValueError(f"Required column '{target_column}' not found in the dataset")

# Normalize case column if present
case_col = next((col for col in df.columns if col.lower() in ['case', 'caseid']), None)
if case_col and case_col != 'Case':
    df = df.rename(columns={case_col: 'Case'})

# Store original activities
df['original_activity'] = df[target_column]
print(f"Run 1: Unique activities before fixing: {df[target_column].nunique()}")

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
print(f"Run 1: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Detect polluted groups
group_stats = df.groupby('BaseActivity').agg(
    unique_variants=pd.NamedAgg(column='original_activity', aggfunc='nunique'),
    total_count=pd.NamedAgg(column='original_activity', aggfunc='count')
).reset_index()
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants]['BaseActivity'].tolist()
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
    print("✓" if precision >= 0.6 else "✗", "Precision threshold (≥ 0.6) met/not met")
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
df_fixed = df[output_columns]
output_path = f"data/bpic11/bpic11_polluted_cleaned_run1.csv"
df_fixed.to_csv(output_path, index=False)

# Summary statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df)) * 100
activity_reduction = unique_activities_before - unique_activities_after
activity_reduction_pct = (activity_reduction / unique_activities_before) * 100

print("\n=== Summary Statistics ===")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Activity reduction count: {activity_reduction}")
print(f"Activity reduction percentage: {activity_reduction_pct:.2f}%")
print(f"Output file path: {output_path}")

# Sample transformations
changed_samples = df[df['is_polluted_label'] == 1][['original_activity', target_column]].drop_duplicates().head(10)
print("\nSample transformations (original → fixed):")
for _, row in changed_samples.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

# Final prints
print(f"\nRun 1: Processed dataset saved to: {output_path}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: {dataset_name}")
print(f"Run 1: Task type: polluted")