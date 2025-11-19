# Generated script for Credit-Polluted - Run 1
# Generated on: 2025-11-18T21:13:14.017926
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run1.csv'
dataset_name = 'credit'
run_number = 1
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")
if 'Activity' not in df.columns:
    raise ValueError("Required column 'Activity' missing.")
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})
df['original_activity'] = df['Activity'].copy()
before_unique = df['Activity'].nunique()
print(f"Number of unique activities before fixing: {before_unique}")

# Define Aggressive Normalization
def aggressive_normalize(activity, token_limit=3):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    # Replace punctuation with spaces
    activity = re.sub(r'[ _\-.,;:]', ' ', activity)
    # Remove long digit strings
    activity = re.sub(r'\d{5,}', '', activity)
    # Split into tokens and filter those without digits
    tokens = activity.split()
    filtered_tokens = [t for t in tokens if not re.search(r'\d', t)]
    # Limit tokens
    cleaned_tokens = filtered_tokens[:token_limit]
    return ' '.join(cleaned_tokens).strip()

# Apply Normalization
df['BaseActivity'] = df['Activity'].apply(lambda x: aggressive_normalize(x, aggressive_token_limit))
initial_rows = len(df)
df = df[df['BaseActivity'] != ''].copy()
dropped = initial_rows - len(df)
if dropped > 0:
    print(f"Dropped {dropped} rows with empty BaseActivity")
print(f"Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect Polluted Groups
grouped = df.groupby('BaseActivity').agg(
    unique_variants=('Activity', 'nunique'),
    total_count=('Activity', 'count')
).reset_index()
polluted_bases = grouped[grouped['unique_variants'] > min_variants]['BaseActivity'].tolist()
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
if 'label' in df.columns:
    y_true = ((df['label'].notna()) & (df['label'].astype(str).str.strip() != '')).astype(int)
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
    precision = recall = f1 = 0.0000
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("No ground-truth labels found, skipping evaluation")

# Integrity Check
print(f"Total polluted bases detected: {len(polluted_bases)}")
print(f"Total events flagged as polluted: {polluted_count}")
print(f"Total clean events: {clean_count}")

# Fix Activities
df.loc[df['is_polluted_label'] == 1, 'Activity'] = df['BaseActivity']

# Summary Statistics
after_unique = df['Activity'].nunique()
reduction_count = before_unique - after_unique
reduction_pct = (reduction_count / before_unique * 100) if before_unique > 0 else 0
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {polluted_count}")
replacement_rate = (polluted_count / len(df) * 100) if len(df) > 0 else 0
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {before_unique} → {after_unique}")
print(f"Activity reduction count and percentage: {reduction_count} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")
# Sample transformations
print("Sample transformations (up to 10):")
changed = df[df['is_polluted_label'] == 1][['original_activity', 'Activity']].head(10)
if len(changed) == 0:
    print("No transformations applied.")
else:
    for _, row in changed.iterrows():
        print(f"{row['original_activity']} → {row['Activity']}")

# Save Fixed Output
helper_columns = ['original_activity', 'BaseActivity', 'is_polluted_label']
df_fixed = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')
df_fixed.to_csv(output_file, index=False)
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {df_fixed.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: polluted")