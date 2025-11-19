# Generated script for Credit-Polluted - Run 2
# Generated on: 2025-11-18T21:13:49.959942
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run2.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
run_number = 2
dataset_name = 'credit'

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found.")

# Normalize column names for Case/CaseID if present
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Store original activities
df['original_activity'] = df[target_column].copy()
unique_before = df[target_column].nunique()
print(f"Number of unique activities before fixing: {unique_before}")

# Define Aggressive Normalization Function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    act = str(activity).lower()
    # Replace punctuation with spaces
    act = re.sub(r'[ _\-. ,;:]', ' ', act)
    # Collapse whitespace
    act = re.sub(r'\s+', ' ', act).strip()
    # Split into tokens and remove ID-like tokens containing digits
    tokens = act.split()
    cleaned_tokens = [t for t in tokens if not re.search(r'\d', t)]
    # Limit tokens
    limited_tokens = cleaned_tokens[:aggressive_token_limit]
    # Join and return
    return ' '.join(limited_tokens)

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)

# Drop rows with empty BaseActivity
initial_len = len(df)
df = df[df['BaseActivity'] != ''].copy()
dropped_count = initial_len - len(df)
if dropped_count > 0:
    print(f"Dropped {dropped_count} rows with empty BaseActivity.")
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
grouped = df.groupby('BaseActivity')['original_activity'].nunique()
polluted_bases = grouped[grouped > min_variants].index.tolist()
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = 0
df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'] = 1
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
print("\n=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average='binary')
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation")

# Integrity Check
print("\nIntegrity Check:")
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")
flagged_in_polluted = df[df['BaseActivity'].isin(polluted_bases)]['is_polluted_label'].sum()
print(f"Events in polluted bases flagged: {flagged_in_polluted} (should equal polluted events)")

# Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Sample Transformations
print("\nSample transformations (up to 10):")
changed = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10)
if len(changed) == 0:
    print("No transformations applied.")
else:
    for _, row in changed.iterrows():
        print(f"{row['original_activity']} → {row[target_column']}")

# Summary Statistics
print("\nSummary Statistics:")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {polluted_count}")
replacement_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Replacement rate: {replacement_rate:.2f}%")
unique_after = df[target_column].nunique()
print(f"Unique activities before: {unique_before} → after: {unique_after}")
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Save Fixed Output
helpers = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helpers if col in df.columns], errors='ignore')
df_fixed.to_csv(output_file, index=False)

# Final Prints
print(f"\nRun {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: polluted")