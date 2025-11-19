# Generated script for BPIC15-Polluted - Run 2
# Generated on: 2025-11-18T21:53:42.031801
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-Polluted.csv'
output_file = 'data/bpic15/bpic15_polluted_cleaned_run2.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
target_column = 'Activity'
label_column = 'label'

# Run identifier prints
print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")
print("This is run #2 of the process")

# Load and Validate
df = pd.read_csv(input_file)
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' missing.")
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})
df['original_activity'] = df[target_column]
print(f"Original dataset shape: {df.shape}")
unique_before = df[target_column].nunique()
print(f"Number of unique activities before fixing: {unique_before}")

# Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=3):
    if pd.isna(activity) or activity == '':
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[_.\-,:;]', ' ', activity)
    activity = re.sub(r'\d{5,}', '', activity)
    tokens = activity.split()
    cleaned_tokens = [token for token in tokens if not re.search(r'\d', token)]
    activity = ' '.join(cleaned_tokens).strip()
    tokens = activity.split()[:token_limit]
    activity = ' '.join(tokens)
    return activity

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize, token_limit=aggressive_token_limit)
initial_rows = len(df)
df = df[df['BaseActivity'] != ''].copy()
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=(target_column, 'nunique'),
    total_count=(target_column, 'count')
).reset_index()
polluted_bases = set(grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].values)
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
pollution_rate = (polluted_count / len(df)) * 100 if len(df) > 0 else 0
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
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
    precision = recall = f1 = 0.0000
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("✗ Precision threshold (≥ 0.6) not met")

# Integrity Check
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Summary Statistics
unique_after = df[target_column].nunique()
reduction = unique_before - unique_after
reduction_perc = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {pollution_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({reduction_perc:.2f}%)")
print(f"Output file path: {output_file}")
print("Sample transformations:")
changed_rows = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10)
if len(changed_rows) > 0:
    for _, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")
else:
    print("No transformations applied.")

# Save Fixed Output
df = df.drop(columns=[col for col in ['original_activity', 'BaseActivity', 'is_polluted_label'] if col in df.columns])
df.to_csv(output_file, index=False)

# Final Run Prints
print(f"Run 2: Processed dataset saved to: data/bpic15/bpic15_polluted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: polluted")