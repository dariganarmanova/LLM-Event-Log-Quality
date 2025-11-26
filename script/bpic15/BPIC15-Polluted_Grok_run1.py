# Generated script for BPIC15-Polluted - Run 1
# Generated on: 2025-11-18T21:52:58.789898
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-Polluted.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
task_type = 'polluted'
run_number = 1
output_file = f'data/bpic15/{dataset_name}_{task_type}_cleaned_run{run_number}.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if target_column not in df.columns:
    raise ValueError(f"{target_column} column is missing")

case_column = 'Case'
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

df['original_activity'] = df[target_column].copy()

print(f"Number of unique activities before fixing: {df[target_column].nunique()}")

# Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=3):
    if pd.isna(activity):
        return ''
    act = str(activity).lower()
    # Replace punctuation with spaces
    act = re.sub(r'[_.,;:\-]', ' ', act)
    # Remove alphanumeric ID-like tokens (tokens containing digits)
    tokens = act.split()
    cleaned_tokens = [token for token in tokens if not re.search(r'\d', token)]
    act = ' '.join(cleaned_tokens)
    # Remove long digit strings (5+ digits)
    act = re.sub(r'\d{5,}', '', act)
    # Collapse whitespace
    act = re.sub(r'\s+', ' ', act).strip()
    # Token limiting
    tokens = act.split()
    limited_tokens = tokens[:token_limit]
    return ' '.join(limited_tokens)

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(lambda x: aggressive_normalize(x, aggressive_token_limit))
print(f"Unique base activities count: {df[df['BaseActivity'] != '']['BaseActivity'].nunique()}")

# Detect Polluted Groups
grouped = df[df['BaseActivity'] != ''][['BaseActivity', 'original_activity']].groupby('BaseActivity')
polluted_bases = set()
for base, group in grouped:
    unique_variants = group['original_activity'].nunique()
    if unique_variants > min_variants:
        polluted_bases.add(base)
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = 0
mask = (df['BaseActivity'] != '') & (df['BaseActivity'].isin(polluted_bases))
df.loc[mask, 'is_polluted_label'] = 1

polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
total = len(df)
pollution_rate = (polluted_count / total * 100) if total > 0 else 0
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
else:
    precision = 0.0000
    recall = 0.0000
    f1 = 0.0000
    print("No ground-truth labels found, skipping evaluation")

print("=== Detection Performance Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
if precision >= 0.6:
    print("✓ Precision threshold (≥ 0.6) met")
else:
    print("✗ Precision threshold (≥ 0.6) not met")

# Integrity Check
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# Fix Activities
mask_fix = df['is_polluted_label'] == 1
df.loc[mask_fix, target_column] = df.loc[mask_fix, 'BaseActivity']

# Summary Statistics
unique_before = df['original_activity'].nunique()
unique_after = df[target_column].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0.0

print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {polluted_count / len(df) * 100:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
print("Sample transformations:")
changed_rows = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10)
for _, row in changed_rows.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

# Save Fixed Output
helper_columns = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helper_columns if col in df.columns])
df_fixed.to_csv(output_file, index=False)

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")