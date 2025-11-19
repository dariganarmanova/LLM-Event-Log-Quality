# Generated script for Pub-Polluted - Run 3
# Generated on: 2025-11-18T18:48:01.802538
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/pub/Pub-Polluted.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
task_type = 'polluted'
output_file = 'data/pub/pub_polluted_cleaned_run3.csv'
run_number = 3
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load and Validate
df = pd.read_csv(input_file)
original_shape = df.shape
print(f"Run {run_number}: Original dataset shape: {original_shape}")

if target_column not in df.columns:
    raise ValueError(f"Missing required column: {target_column}")

# Normalize column names for Case
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Store original activities
df['original_activity'] = df[target_column].copy()

# Print unique activities before fixing
unique_before = df[target_column].nunique()
print(f"Number of unique activities before fixing: {unique_before}")

# Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=3):
    if pd.isna(activity):
        return ''
    # 1. Lowercase
    activity = str(activity).lower()
    # 2. Replace punctuation with spaces
    activity = re.sub(r'[_\\-.,;:]', ' ', activity)
    # 4. Remove long digit strings (sequences of 5+ digits)
    activity = re.sub(r'\d{5,}', '', activity)
    # 3. Remove alphanumeric ID-like tokens: split and filter tokens containing digits
    tokens = activity.split()
    cleaned_tokens = [token for token in tokens if not re.search(r'\d', token)]
    activity = ' '.join(cleaned_tokens)
    # 5. Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity).strip()
    # 6. Token limiting: keep only the first token_limit tokens
    tokens = activity.split()
    limited_tokens = tokens[:token_limit]
    activity = ' '.join(limited_tokens)
    # 7. Return
    return activity

# Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize, token_limit=aggressive_token_limit)

# Print unique base activities count (non-empty)
non_empty_bases = df[df['BaseActivity'] != '']['BaseActivity']
print(f"Unique base activities count: {non_empty_bases.nunique()}")

# Detect Polluted Groups
df_nonempty = df[df['BaseActivity'] != '']
grouped = df_nonempty.groupby('BaseActivity').agg(
    unique_variants=('Activity', 'nunique'),
    total_count=('Activity', 'count')
)
polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag Polluted Events
df['is_polluted_label'] = 0
mask = df['BaseActivity'].isin(polluted_bases)
df.loc[mask, 'is_polluted_label'] = 1
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {polluted_count / len(df) * 100:.2f}%")

# Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
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
unique_after = df_fixed[target_column].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
replacement_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations (up to 10, only changed rows)
print("Sample transformations:")
changed_mask = df['is_polluted_label'] == 1
changed_df = df[changed_mask][['original_activity', target_column]].head(10)
if len(changed_df) == 0:
    print("No transformations applied.")
else:
    for _, row in changed_df.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")

# Final prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")