# Generated script for Credit-Polluted - Run 3
# Generated on: 2025-11-13T16:31:12.797171
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Polluted.csv'
output_file = 'data/credit/credit_polluted_cleaned_run3.csv'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
case_column = 'Case'
timestamp_column = 'Timestamp'
resource_column = 'Resource' if 'Resource' in pd.read_csv(input_file, nrows=1).columns else None
variant_column = 'Variant' if 'Variant' in pd.read_csv(input_file, nrows=1).columns else None

# Step 1: Load and Validate
df = pd.read_csv(input_file)
df.columns = [col.replace('CaseID', 'Case') for col in df.columns]  # Normalize Case/CaseID
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the input file.")
df['original_activity'] = df[target_column]
print(f"Run 3: Original dataset shape: {df.shape}")
print(f"Run 3: Unique activities before fixing: {df[target_column].nunique()}")

# Step 2: Define Aggressive Normalization
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Step 1: Lowercase
    s = str(activity).lower()
    # Step 2: Replace punctuation with spaces
    s = re.sub(r'[_\-\.,;:]', ' ', s)
    # Step 3: Remove alphanumeric ID-like tokens
    s = ' '.join([token for token in s.split() if not any(c.isdigit() for c in token)])
    # Step 4: Remove long digit strings (5+ digits)
    s = re.sub(r'\b\d{5,}\b', '', s)
    # Step 5: Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Step 6: Token limiting
    tokens = s.split()[:aggressive_token_limit]
    return ' '.join(tokens)

# Step 3: Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ''].copy()  # Drop rows with empty BaseActivity
print(f"Run 3: Unique base activities count: {df['BaseActivity'].nunique()}")

# Step 4: Detect Polluted Groups
grouped = df.groupby('BaseActivity')[target_column].agg(['nunique', 'count']).rename(columns={'nunique': 'unique_variants', 'count': 'total_count'})
polluted_bases = grouped[grouped['unique_variants'] > min_variants].index.tolist()
print(f"Run 3: Number of polluted groups found: {len(polluted_bases)}")

# Step 5: Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 3: Polluted events count: {polluted_count}")
print(f"Run 3: Clean events count: {clean_count}")
print(f"Run 3: Pollution rate: {pollution_rate:.2f}%")

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (~df[label_column].isna() & (df[label_column] != '')).astype(int)
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation")
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Step 7: Integrity Check
print(f"Run 3: Integrity check - Total polluted bases detected: {len(polluted_bases)}")
print(f"Run 3: Integrity check - Total events flagged as polluted: {polluted_count}")
print(f"Run 3: Integrity check - Total clean events: {clean_count}")

# Step 8: Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Step 9: Save Fixed Output
columns_to_drop = ['original_activity', 'BaseActivity', 'is_polluted_label']
columns_to_keep = [col for col in df.columns if col not in columns_to_drop]
df_fixed = df[columns_to_keep].copy()
df_fixed.to_csv(output_file, index=False)

# Step 11: Summary Statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df_fixed[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df_fixed)) * 100
activity_reduction = unique_activities_before - unique_activities_after
activity_reduction_pct = (activity_reduction / unique_activities_before) * 100

print(f"\nRun 3: Summary Statistics")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df_fixed)}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Activity reduction count and percentage: {activity_reduction} ({activity_reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
changed_rows = df[df['is_polluted_label'] == 1].head(10)
if not changed_rows.empty:
    print("\nSample transformations (original_activity → Activity):")
    for _, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")
else:
    print("\nNo polluted activities were changed.")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_fixed.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: polluted")