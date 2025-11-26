# Generated script for BPIC11-Polluted - Run 1
# Generated on: 2025-11-18T22:40:01.650818
# Model: grok-4-fast

import pandas as pd
import re

# Configuration
input_file = 'data/bpic11/BPIC11-Polluted.csv'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run1.csv'
dataset_name = 'bpic11'
task_type = 'polluted'
run_number = 1
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
target_column = 'Activity'
label_column = 'label'

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' is missing.")

# Normalize column names for Case if needed
possible_case_names = ['Case ID', 'case', 'case id']
for pc in possible_case_names:
    if pc in df.columns and target_column in df.columns:
        df.rename(columns={pc: 'Case'}, inplace=True)
        break

# Store original activities
df['original_activity'] = df[target_column]
num_unique_before = df[target_column].nunique()
print(f"Number of unique activities before fixing: {num_unique_before}")

# Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=aggressive_token_limit):
    if pd.isna(activity):
        return ''
    act = str(activity).lower()
    # Step 2: Replace punctuation with spaces
    act = re.sub(r'[_ \-.,;:]', ' ', act)
    # Step 4: Remove long digit strings (replace with space to split)
    act = re.sub(r'\d{5,}', ' ', act)
    # Step 3: Split into tokens and remove tokens containing digits
    tokens = act.split()
    clean_tokens = [t for t in tokens if not re.search(r'\d', t) and t.strip()]
    # Step 6: Limit tokens
    limited_tokens = clean_tokens[:token_limit]
    # Step 5: Collapse whitespace (already handled by join)
    return ' '.join(limited_tokens).strip()

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
# Drop rows with empty BaseActivity
initial_rows = len(df)
df = df[df['BaseActivity'] != ''].copy()
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('original_activity', 'nunique'),
    total_count=('original_activity', 'count')
).reset_index()
polluted_bases = set(grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist())
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
total_after_drop = len(df)
pollution_rate = (polluted_count / total_after_drop * 100) if total_after_drop > 0 else 0
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
def compute_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    prec, rec, f1 = compute_metrics(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
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

# Summary Statistics
labels_replaced = polluted_count
replacement_rate = pollution_rate
unique_after = df[target_column].nunique()
reduction = num_unique_before - unique_after
reduction_pct = (reduction / num_unique_before * 100) if num_unique_before > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {labels_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {num_unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed_mask = df['is_polluted_label'] == 1
changed_rows = df[changed_mask][['original_activity', target_column]].head(10)
print("Sample transformations (original → new):")
if len(changed_rows) > 0:
    for _, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")
else:
    print("No transformations applied.")

# Save Fixed Output
helper_columns = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helper_columns if col in df.columns])
df_fixed.to_csv(output_file, index=False)

# Final prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")