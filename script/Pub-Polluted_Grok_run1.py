# Generated script for Pub-Polluted - Run 1
# Generated on: 2025-11-18T18:47:03.763495
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Polluted.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
output_file = 'data/pub/pub_polluted_cleaned_run1.csv'
run_number = 1
task_type = 'polluted'
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
    raise ValueError(f"Required column '{target_column}' missing.")

# Normalize column names for Case/CaseID
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Store original activities
df['original_activity'] = df[target_column]

# Print unique activities before
print(f"Number of unique activities before fixing: {df[target_column].nunique()}")

# Define Aggressive Normalization
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    # Replace punctuation with spaces
    activity = re.sub(r'[_ \-.,;:]', ' ', activity)
    # Remove long digit strings (5+ digits)
    activity = re.sub(r'\d{5,}', '', activity)
    # Split into tokens
    tokens = activity.split()
    # Remove alphanumeric ID-like tokens containing digits
    tokens = [t for t in tokens if not re.search(r'\d', t)]
    # Join and limit tokens
    activity = ' '.join(tokens[:aggressive_token_limit]).strip()
    return activity

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)

# Print unique base activities count (non-empty)
non_empty_bases_count = (df['BaseActivity'] != '').sum()
unique_base_count = df.loc[df['BaseActivity'] != '', 'BaseActivity'].nunique()
print(f"Unique base activities count: {unique_base_count}")

# Detect Polluted Groups
df_noempty = df[df['BaseActivity'] != ''].copy()
if len(df_noempty) > 0:
    grouped = df_noempty.groupby('BaseActivity').agg(
        unique_variants=(target_column, 'nunique'),
        total_count=(target_column, 'count')
    )
    polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
else:
    polluted_bases = []
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = 0
mask_polluted = df['BaseActivity'].isin(polluted_bases)
df.loc[mask_polluted, 'is_polluted_label'] = 1

polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
pollution_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column'].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Save Fixed Output
helper_columns = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helper_columns if col in df.columns])
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
labels_replaced = polluted_count
replacement_rate = (labels_replaced / len(df) * 100) if len(df) > 0 else 0
print(f"Labels replaced count: {labels_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
unique_before = df['original_activity'].nunique()
unique_after = df_fixed[target_column].nunique()
print(f"Unique activities before → after: {unique_before} → {unique_after}")
reduction_count = unique_before - unique_after
reduction_pct = (reduction_count / unique_before * 100) if unique_before > 0 else 0
print(f"Activity reduction count and percentage: {reduction_count} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations (up to 10 for changed rows)
print("Sample transformations:")
changed_sample = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10)
if len(changed_sample) > 0:
    for _, row in changed_sample.iterrows():
        print(f"  {row['original_activity']} → {row[target_column]}")
else:
    print("  No transformations applied.")

# Final prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")