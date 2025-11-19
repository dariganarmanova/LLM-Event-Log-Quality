# Generated script for Pub-Polluted - Run 2
# Generated on: 2025-11-18T18:47:32.100306
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Polluted.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
output_file = 'data/pub/pub_polluted_cleaned_run2.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False
run_number = 2

# Load the data
df = pd.read_csv(input_file)

# Normalize column names for Case
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': 'Case'})

print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Ensure required column exists
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the dataset.")

# Store original activities
df['original_activity'] = df[target_column].copy()

# Print unique activities before fixing
print(f"Number of unique activities before fixing: {df[target_column].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    act = str(activity).lower()
    # Step 2: Replace punctuation with spaces
    act = re.sub(r'[_\-\.\,\;\:]', ' ', act)
    # Step 4: Remove long digit strings (5+ digits)
    act = re.sub(r'\b\d{5,}\b', '', act)
    # Step 3: Split and remove alphanumeric ID-like tokens (containing digits)
    tokens = act.split()
    tokens = [t for t in tokens if not re.search(r'\d', t)]
    # Step 5: Collapse whitespace
    act = ' '.join(tokens).strip()
    # Step 6: Token limiting
    tokens = act.split()[:aggressive_token_limit]
    return ' '.join(tokens)

# Step 3: Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)

# Create sub_df for non-empty BaseActivity for detection
non_empty_mask = (df['BaseActivity'] != '') & df['BaseActivity'].notna()
sub_df = df[non_empty_mask].copy()
print(f"Unique base activities count: {sub_df['BaseActivity'].nunique()}")

# Step 4: Detect Polluted Groups
grouped = sub_df.groupby('BaseActivity')
unique_variants = grouped['original_activity'].nunique()
total_count = grouped.size()
polluted_bases = [base for base, vars_count in unique_variants.items() if vars_count > min_variants]
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Step 5: Flag Polluted Events
df['is_polluted_label'] = 0
df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'] = 1
polluted_events = df['is_polluted_label'].sum()
clean_events = len(df) - polluted_events
pollution_rate = (polluted_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Polluted events count: {polluted_events}")
print(f"Clean events count: {clean_events}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No ground-truth labels found, skipping evaluation")
precision_met = "✓" if precision >= 0.6 else "✗"
print(f"{precision_met} Precision threshold (>= 0.6) {'met' if precision >= 0.6 else 'not met'}")

# Step 7: Integrity Check
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_events}")
print(f"Total clean events: {clean_events}")
# Verify only events in polluted bases are flagged
flagged_in_polluted = df[df['BaseActivity'].isin(polluted_bases)]['is_polluted_label'].sum()
assert flagged_in_polluted == polluted_events, "Integrity check failed: Mismatch in flagged events."

# Step 8: Fix Activities
changed_mask = df['is_polluted_label'] == 1
df.loc[changed_mask, target_column] = df.loc[changed_mask, 'BaseActivity']

# Step 9: Save Fixed Output
helper_columns = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_rows = len(df)
replaced_count = changed_mask.sum()
replacement_rate = (replaced_count / total_rows * 100) if total_rows > 0 else 0
before_unique = df['original_activity'].nunique()
after_unique = df_fixed[target_column].nunique()
reduction_count = before_unique - after_unique
reduction_pct = (reduction_count / before_unique * 100) if before_unique > 0 else 0

print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {total_rows}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate (%): {replacement_rate:.2f}")
print(f"Unique activities before → after: {before_unique} → {after_unique}")
print(f"Activity reduction count and percentage: {reduction_count} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations (up to 10 changed rows)
print("Sample transformations (original_activity → Activity):")
changed_samples = df[changed_mask].head(10)
for _, row in changed_samples.iterrows():
    print(f"{row['original_activity']} → {row[target_column]}")

# Required run prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: polluted")