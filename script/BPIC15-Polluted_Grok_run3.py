# Generated script for BPIC15-Polluted - Run 3
# Generated on: 2025-11-18T21:54:21.466218
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic15/BPIC15-Polluted.csv'
output_file = 'data/bpic15/bpic15_polluted_cleaned_run3.csv'
dataset_name = 'bpic15'
normalization_strategy = 'aggressive'
aggressive_token_limit = 3
min_variants = 2
label_column = 'label'

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load data
df = pd.read_csv(input_file)

# Validate required column
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' not found.")

# Normalize case column if needed
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Store original activities
df['original_activity'] = df['Activity']

# Print baseline
print(f"Original dataset shape: {df.shape}")
unique_before = df['Activity'].nunique()
print(f"Number of unique activities before fixing: {unique_before}")

# Define aggressive normalization function
def aggressive_normalize(activity, aggressive_token_limit=3):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    # Replace punctuation with spaces
    activity = re.sub(r'[_.\-,;:]', ' ', activity)
    # Remove long digit strings (5+ digits)
    activity = re.sub(r'\d{5,}', '', activity)
    # Split and remove tokens containing any digits
    tokens = [t for t in activity.split() if not re.search(r'\d', t)]
    activity = ' '.join(tokens)
    # Limit tokens
    tokens = activity.split()[:aggressive_token_limit]
    activity = ' '.join(tokens)
    return activity.strip()

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize, args=(aggressive_token_limit,))

# Print unique base activities (non-empty)
unique_bases = df[df['BaseActivity'] != '']['BaseActivity'].nunique()
print(f"Unique base activities count: {unique_bases}")

# Detect polluted groups using non-empty BaseActivity
non_empty_df = df[df['BaseActivity'] != '']
if len(non_empty_df) > 0:
    grouped = non_empty_df.groupby('BaseActivity')['Activity'].agg(['nunique', 'count']).rename(columns={'nunique': 'unique_variants', 'count': 'total_count'})
    polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
else:
    polluted_bases = []
print(f"Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = 0
df.loc[df['BaseActivity'].isin(polluted_bases), 'is_polluted_label'] = 1
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Polluted events count: {polluted_count}")
print(f"Clean events count: {clean_count}")
print(f"Pollution rate: {pollution_rate:.2f}%")

# Integrity check prints
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# Detection metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].fillna('').astype(str).str.strip() != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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

# Fix activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Prepare fixed dataframe
df_fixed = df.drop(columns=['original_activity', 'BaseActivity', 'is_polluted_label'])

# Summary statistics
unique_after = df_fixed['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {polluted_count}")
print(f"Replacement rate: {polluted_count / len(df_fixed) * 100:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
print("Sample transformations (up to 10):")
changed_rows = df[df['is_polluted_label'] == 1][['original_activity', 'Activity']].head(10)
for idx, row in changed_rows.iterrows():
    print(f"{row['original_activity']} → {row['Activity']}")

# Save
df_fixed.to_csv(output_file, index=False)
print(f"Run 3: Processed dataset saved to: data/bpic15/bpic15_polluted_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: polluted")