# Generated script for BPIC11-Polluted - Run 3
# Generated on: 2025-11-18T22:41:03.699371
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic11/BPIC11-Polluted.csv'
input_directory = 'data/bpic11'
dataset_name = 'bpic11'
output_file = 'data/bpic11/bpic11_polluted_cleaned_run3.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# #1. Load and Validate
df = pd.read_csv(input_file)

# Normalize column names for Case
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': 'Case'})
if 'case id' in df.columns:
    df = df.rename(columns={'case id': 'Case'})
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Ensure required column exists
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' is missing.")

# Store original activities
df['original_activity'] = df[target_column].copy()

# Print baseline
print(f"Original dataset shape: {df.shape}")
print(f"Number of unique activities before fixing: {df[target_column].nunique()}")

# #2. Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=3):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    # Replace punctuation with spaces
    activity = re.sub(r'[ _\-.,;:]', ' ', activity)
    # Remove long digit strings (5+ digits)
    activity = re.sub(r'\b\d{5,}\b', '', activity)
    # Split into tokens and remove tokens containing digits
    tokens = activity.split()
    cleaned_tokens = [token for token in tokens if not re.search(r'\d', token)]
    activity = ' '.join(cleaned_tokens)
    # Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity).strip()
    # Limit to first N tokens
    tokens = activity.split()
    limited_tokens = tokens[:token_limit]
    return ' '.join(limited_tokens)

# #3. Apply Normalization
df['BaseActivity'] = df[target_column].apply(lambda x: aggressive_normalize(x, aggressive_token_limit))
initial_rows = len(df)
df = df[df['BaseActivity'] != ''].copy()
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# #4. Detect Polluted Groups
grouped = df.groupby('BaseActivity').agg({
    target_column: 'nunique',
    'original_activity': 'size'
}).rename(columns={target_column: 'unique_variants', 'original_activity': 'total_count'})

polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
print(f"Number of polluted groups found: {len(polluted_bases)}")

# #5. Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
total_after_drop = len(df)
pollution_rate = (polluted_count / total_after_drop * 100) if total_after_drop > 0 else 0
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# #6. Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns and df[label_column].notna().any():
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
    y_pred = df['is_polluted_label']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
else:
    print("No ground-truth labels found, skipping evaluation")
    prec = rec = f1 = 0.0000
print(f"=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
if prec >= 0.6:
    print("✓ Precision threshold (≥ 0.6) met")
else:
    print("✗ Precision threshold (≥ 0.6) not met")

# #7. Integrity Check
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# #8. Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Collect sample transformations for changed rows
changed_rows = df[df['is_polluted_label'] == 1][['original_activity', target_column]].head(10).values
print("Sample transformations (original → fixed):")
for orig, fixed in changed_rows:
    print(f"{orig} → {fixed}")

# #9. Save Fixed Output
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'], errors='ignore')
df_fixed.to_csv(output_file, index=False)

# #11. Summary Statistics
unique_before = df['original_activity'].nunique()
unique_after = df_fixed[target_column].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
replacement_rate = (polluted_count / initial_rows * 100) if initial_rows > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 3: Processed dataset saved to: data/bpic11/bpic11_polluted_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: polluted")