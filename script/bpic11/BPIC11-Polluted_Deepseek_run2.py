# Generated script for BPIC11-Polluted - Run 2
# Generated on: 2025-11-13T11:47:26.352351
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Polluted.csv'
dataset_name = 'bpic11'
output_suffix = '_polluted_cleaned_run2'
target_column = 'Activity'
label_column = 'label'
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Step 1: Load and Validate
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure required columns exist
if target_column not in df.columns:
    raise ValueError(f"Required column '{target_column}' not found in the dataset")

# Store original activities
df['original_activity'] = df[target_column].copy()
print(f"Run 2: Unique activities before fixing: {df[target_column].nunique()}")

# Step 2: Define Aggressive Normalization
def aggressive_normalize(activity):
    if pd.isna(activity):
        return ''
    # Lowercase
    s = str(activity).lower()
    # Replace punctuation with spaces
    s = re.sub(r'[_\-\.,;:]', ' ', s)
    # Remove tokens containing digits
    s = ' '.join([token for token in s.split() if not any(c.isdigit() for c in token)])
    # Remove long digit strings (5+ digits)
    s = re.sub(r'\d{5,}', '', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    # Token limiting
    tokens = s.split()[:aggressive_token_limit]
    return ' '.join(tokens).strip()

# Step 3: Apply Normalization
df['BaseActivity'] = df[target_column].apply(aggressive_normalize)
df = df[df['BaseActivity'] != ''].copy()
print(f"Run 2: Unique base activities after normalization: {df['BaseActivity'].nunique()}")

# Step 4: Detect Polluted Groups
group_stats = df.groupby('BaseActivity').agg(
    unique_variants=pd.NamedAgg(column='original_activity', aggfunc='nunique'),
    total_count=pd.NamedAgg(column='original_activity', aggfunc='count')
).reset_index()
polluted_bases = group_stats[group_stats['unique_variants'] > min_variants]['BaseActivity'].tolist()
print(f"Run 2: Number of polluted groups found: {len(polluted_bases)}")

# Step 5: Flag Polluted Events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases).astype(int)
polluted_count = df['is_polluted_label'].sum()
clean_count = len(df) - polluted_count
pollution_rate = (polluted_count / len(df)) * 100
print(f"Run 2: Polluted events count: {polluted_count}")
print(f"Run 2: Clean events count: {clean_count}")
print(f"Run 2: Pollution rate: {pollution_rate:.2f}%")

# Step 6: Calculate Detection Metrics
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
    print("✓" if precision >= 0.6 else "✗", "Precision threshold (≥ 0.6) met/not met")
else:
    print("No ground-truth labels found, skipping evaluation")
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Step 7: Integrity Check
print(f"Run 2: Total polluted bases detected: {len(polluted_bases)}")
print(f"Run 2: Total events flagged as polluted: {polluted_count}")
print(f"Run 2: Total clean events: {clean_count}")

# Step 8: Fix Activities
df.loc[df['is_polluted_label'] == 1, target_column] = df.loc[df['is_polluted_label'] == 1, 'BaseActivity']

# Step 9: Save Fixed Output
output_columns = [col for col in df.columns if col not in ['original_activity', 'BaseActivity', 'is_polluted_label']]
df_fixed = df[output_columns].copy()
output_path = f"data/bpic11/{dataset_name}{output_suffix}.csv"
df_fixed.to_csv(output_path, index=False)

# Step 11: Summary Statistics
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df[target_column].nunique()
replaced_count = polluted_count
replacement_rate = (replaced_count / len(df)) * 100
activity_reduction = unique_activities_before - unique_activities_after
reduction_percentage = (activity_reduction / unique_activities_before) * 100

print("\n=== Summary Statistics ===")
print(f"Normalization strategy: {normalization_strategy}")
print(f"Total rows: {len(df)}")
print(f"Labels replaced count: {replaced_count}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Activity reduction count: {activity_reduction}")
print(f"Activity reduction percentage: {reduction_percentage:.2f}%")
print(f"Output file path: {output_path}")

# Print sample transformations
changed_rows = df[df['is_polluted_label'] == 1].head(10)
if not changed_rows.empty:
    print("\nSample transformations (original → fixed):")
    for _, row in changed_rows.iterrows():
        print(f"{row['original_activity']} → {row[target_column]}")

# Required final prints
print(f"\nRun 2: Processed dataset saved to: {output_path}")
print(f"Run 2: Final dataset shape: {df_fixed.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: polluted")