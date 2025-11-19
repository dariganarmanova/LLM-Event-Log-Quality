# Generated script for Credit-Polluted - Run 3
# Generated on: 2025-11-18T21:14:18.423453
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run3.csv'
dataset_name = 'credit'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'

# Load and validate
df = pd.read_csv(input_file)
if 'Activity' not in df.columns:
    raise ValueError("Activity column missing")
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})
df['original_activity'] = df['Activity'].copy()
print(f"Run 3: Original dataset shape: {df.shape}")
print(f"Run 3: Number of unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[_.\-,;:]', ' ', activity)
    tokens = activity.split()
    cleaned_tokens = [token for token in tokens if not re.search(r'\d', token)]
    activity = ' '.join(cleaned_tokens).strip()
    activity = re.sub(r'\d{5,}', '', activity)
    tokens = activity.split()[:aggressive_token_limit]
    return ' '.join(tokens)

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df_nonempty = df[df['BaseActivity'] != '']
print(f"Run 3: Unique base activities count: {df_nonempty['BaseActivity'].nunique()}")

# Detect polluted groups
grouped = df_nonempty.groupby('BaseActivity')
unique_variants = grouped['original_activity'].nunique()
polluted_bases = [base for base, var in unique_variants.items() if var > min_variants]
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = 0
df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'] = 1
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
total_rows = len(df)
pollution_rate = (polluted_count / total_rows * 100) if total_rows > 0 else 0
print(f"Run 3: Polluted events count: {polluted_count}")
print(f"Run 3: Clean events count: {clean_count}")
print(f"Run 3: Pollution rate: {pollution_rate:.2f}%")

# Calculate detection metrics (before fixing)
print("=== Detection Performance Metrics ===")
if 'label' in df.columns:
    y_true = (df['label'].notna() & (df['label'].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    met = "✓" if prec >= 0.6 else "✗"
    print(f"{met} Precision threshold (≥ 0.6) met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation")

# Integrity check
print(f"Run 3: Total polluted bases detected: {len(polluted_bases)}")
print(f"Run 3: Total events flagged as polluted: {polluted_count}")
print(f"Run 3: Total clean events: {clean_count}")

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df['BaseActivity']

# Save fixed output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])
df_fixed.to_csv(output_file, index=False)

# Summary statistics
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {pollution_rate:.2f}%")
before_unique = df['original_activity'].nunique()
after_unique = df_fixed['Activity'].nunique()
print(f"Unique activities before → after: {before_unique} → {after_unique}")
reduction_count = before_unique - after_unique
reduction_pct = (reduction_count / before_unique * 100) if before_unique > 0 else 0
print(f"Activity reduction: {reduction_count} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")
changed_rows = df[df['is_polluted_label'] == 1][['original_activity', 'Activity']].head(10)
if len(changed_rows) > 0:
    print("Sample transformations (up to 10):")
    for idx, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row['Activity']}")
else:
    print("No transformations applied.")

# Required summary prints
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: {dataset_name}")
print(f"Run 3: Task type: polluted")