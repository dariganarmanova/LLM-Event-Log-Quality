# Generated script for BPIC15-Polluted - Run 3
# Generated on: 2025-11-13T12:45:03.174142
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-Polluted.csv'
output_file = 'data/bpic15/bpic15_polluted_cleaned_run3.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Step 1: Load and Validate
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the dataset")

# Normalize case column names
case_cols = [col for col in df.columns if col.lower() in ['case', 'caseid']]
if case_cols:
    df = df.rename(columns={case_cols[0]: 'Case'})

df['original_activity'] = df[target_column]
print(f"Run 3: Unique activities before fixing: {df[target_column].nunique()}")

# Step 2: Define Aggressive Normalization
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

# Step 3: Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ''].copy()
print(f"Run 3: Unique base activities: {df['BaseActivity'].nunique()}")

# Step 4: Detect Polluted Groups
group_stats = df.groupby('BaseActivity')[target_column].agg(['nunique', 'count']).reset_index()
group_stats.columns = ['BaseActivity', 'unique_variants', 'total_count']
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 3: Polluted groups found: {len(polluted_bases)}")

# Step 5: Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
print(f"Run 3: Polluted events: {polluted_count}")
print(f"Run 3: Clean events: {clean_count}")
print(f"Run 3: Pollution rate: {polluted_count / len(df) * 100:.2f}%")

# Step 6: Calculate Detection Metrics
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

# Step 7: Integrity Check
print(f"Run 3: Integrity check - Total polluted bases: {len(polluted_bases)}")
print(f"Run 3: Integrity check - Events to be fixed: {polluted_count}")

# Step 8: Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Step 9: Save Fixed Output
output_columns = [col for col in df.columns if col not in ['original_activity', 'BaseActivity', 'is_polluted_label']]
df_fixed = df[output_columns].copy()
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary Statistics
original_unique = df['original_activity'].nunique()
final_unique = df_fixed[target_column].nunique()
replaced_count = polluted_count
replacement_rate = replaced_count / len(df) * 100
reduction_count = original_unique - final_unique
reduction_percent = reduction_count / original_unique * 100

print(f"\nRun 3: Summary Statistics")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {original_unique} → {final_unique}")
print(f"Activity reduction: {reduction_count} ({reduction_percent:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
changed_samples = df[df['is_polluted_label'] == 1][['original_activity', target_column]].drop_duplicates().head(10)
print("\nSample transformations (original → fixed):")
for _, row in changed_samples.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: polluted")