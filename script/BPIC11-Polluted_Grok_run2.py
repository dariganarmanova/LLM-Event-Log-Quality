# Generated script for BPIC11-Polluted - Run 2
# Generated on: 2025-11-18T22:40:33.599709
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support

# Configuration
run_number = 2
dataset_name = 'bpic11'
task_type = 'polluted'
input_file = 'data/bpic11/BPIC11-Polluted.csv'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run2.csv'
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
    raise ValueError(f"{target_column} column missing")

# Normalize column names for Case
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Store original activities
df['original_activity'] = df[target_column]

print(f"Number of unique activities before fixing: {df[target_column].nunique()}")

# Define Aggressive Normalization
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    act = str(activity).lower()
    # Step 2: Replace punctuation with spaces
    act = re.sub(r'[-_.,;:]', ' ', act)
    # Step 4: Remove long digit strings (5+ digits)
    act = re.sub(r'\d{5,}', '', act)
    # Step 3: Split into tokens and remove alphanumeric ID-like tokens (containing digits)
    tokens = act.split()
    tokens = [t for t in tokens if not re.search(r'\d', t)]
    # Step 5: Collapse whitespace (already handled by split/join)
    # Step 6: Token limiting
    tokens = tokens[:aggressive_token_limit]
    # Step 7: Join and trim
    result = ' '.join(tokens).strip()
    return result

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)

# For detection, use non-empty BaseActivity
valid_df = df[df['BaseActivity'] != ''].copy()
print(f"Unique base activities count: {valid_df['BaseActivity'].nunique()}")

# Detect Polluted Groups
group_stats = valid_df.groupby('BaseActivity')[target_column].nunique().reset_index(name='unique_variants')
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = 0
df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'] = 1

polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {polluted_count / len(df) * 100:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = df[label_column].notna() & (df[label_column].fillna('').astype(str).str.strip() != '')
    y_true = y_true.astype(int)
    y_pred = df['is_polluted_label']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
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
mask = df['is_polluted_label'] == 1
df.loc[mask, target_column] = df.loc[mask, 'BaseActivity']

# Save Fixed Output
helper_cols = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop([col for col in helper_cols if col in df.columns], axis=1)
df_fixed.to_csv(output_file, index=False)

# Summary Statistics
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
replaced_count = polluted_count
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replaced_count / len(df) * 100:.2f}%")
unique_before = df['original_activity'].nunique()
unique_after = df_fixed[target_column].nunique()
print(f"Unique activities before → after: {unique_before} → {unique_after}")
reduction = unique_before - unique_after
print(f"Activity reduction count and percentage: {reduction} ({reduction / unique_before * 100:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations (only changed rows, up to 10)
changed_df = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10)
if len(changed_df) > 0:
    print("Sample transformations (original → new):")
    for _, row in changed_df.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")
else:
    print("No transformations applied.")

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")